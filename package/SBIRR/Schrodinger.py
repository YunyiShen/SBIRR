import torch
import pyro.contrib.gp as gp
import math
import gc
import copy
import time
import logging 
import numpy as np
from tqdm import tqdm

from SBIRR.SDE_solver import solve_sde_RK
from SBIRR.GP import spatMultitaskGPModel, MultitaskGPModel, MultitaskGPModelSparse
from SBIRR.NN import Feedforward, train_nn
from SBIRR.RFF import RandomFourierFeatures
from SBIRR.utils import (auxiliary_plot_routine_init, 
                               auxiliary_plot_routine_end)
from SBIRR.unet import get_trained_unet
import memory_profiler

import line_profiler
profile = line_profiler.LineProfiler()

def make_sparseGP(num_data_points,num_time_points, device):
    def func(Xs, Ys, dt, kern, noise, gp_mean_function):
        return MultitaskGPModelSparse(
            Xs, Ys, dt=dt, kern=kern, noise=noise, 
            gp_mean_function=gp_mean_function, num_data_points=num_data_points, 
            num_time_points=num_time_points, device=device) 
    return func


class driftfit:
    '''
    general class for fit drift with AR losses
    '''
    def __init__(self, dt):

        self.dt = dt # Euler time step

    def make_autoregression(self, Xts):
        """
        Split the data into location and finite difference.
        Args:
            Xts: Tensor of shape (num_particles, num_time_steps, non_temporal_dimension + 1)
        Returns:
            Ys: the finite difference of the data (this is the velocity, target of the regression, reshaped
                into (num_particles * (num_time_steps-1), non_temporal_dimension))
            Xs: the spatio-temporal location of data up to the second-to-last time step (this is the input of the regression)
        """
        Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1]) / self.dt).reshape((-1, Xts.shape[2] - 1)) 
        Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) 
        return Ys, Xs
    
    def fit(Xs, Ys, volatility):
        NotImplementedError
    
class baseIPFP:
    def __init__(self, ref_drift=None, 
                 sde_solver = solve_sde_RK, 
                 N=10,
                 device = None):
        if ref_drift is None:
            def ref_drift(x):
                return torch.tensor([0] * (x.shape[1] - 1)).reshape((1, -1)).repeat(x.shape[0], 1).to(device)
            self.ref_drift = ref_drift
        else:
            self.ref_drift = ref_drift # reference drift 
        self.forward_drift = None
        self.backward_drift = None
        self.sde_solver = sde_solver
        self.device = device   
        self.dt = 1./N # Euler time step
        self.device = device  

    def set_ref_drift(self, drift):
        """
        Set the reference drift to learnt drift.
        Args:
            drift: the learnt drift function
        """
        self.ref_drift = drift # reference drift 

    def clear_drifts(self):
        """
        Clear the reference drifts (we might want to reuse the drift for different time chunks)
        """
        self.backward_drift = None
        self.forward_drift = None 
        gc.collect()  


class gpdrift(driftfit):
    '''
    class for drift fitting with GPs
    '''
    
    def __init__(self, kernel = gp.kernels.Exponential, 
                 gpmean = None, 
                 GPfun = MultitaskGPModel,
                 N = 10,
                 device = None):
        driftfit.__init__(self, 1./N)
        self.kernel = kernel
        self.gpmean = gpmean
        self.device = device
        ## see GP.py for how we write one of these gpfuns
        self.GPfun = GPfun # this could be sparse GP with make_sparseGP
    
    def fit(self, Xs, Ys, volatility): # this use auto-regressive objective to fit a drift, used for IPFP
        """
        Learn a drift function using a Gaussian process regression.
        Args:
            Xs: spatio-temporal location of data (dimension num_particles * (num_time_steps-1), non_temporal_dimension + 1)
            Ys: the finite difference of the data (velocity, dimension num_particles * (num_time_steps-1), non_temporal_dimension)
            volatility: the volatility of the SDE
        Returns:
            the fitted drift function
        """
        return self.GPfun(Xs, Ys, dt=self.dt, 
                        kern=self.kernel, noise=volatility ** 2, 
                        gp_mean_function=self.gpmean).predict


class gpIPFP(gpdrift,baseIPFP):
    '''
    class for doing IPFP with GP fitting (forward and backward) drifts
    '''
    def __init__(self, ref_drift=None, 
                 sde_solver = solve_sde_RK,
                 kernel = gp.kernels.Exponential, 
                 gpmean = None, 
                 GPfun = MultitaskGPModel,
                 N=10, device = None):
        gpdrift.__init__(self, kernel, gpmean, GPfun, N, device)
        baseIPFP.__init__(self, ref_drift, sde_solver, N, device)

    def init_ipfp_backward(self, duration, volatility, t0 = 0,
                  prior_X_0=None): #  initialize IPFP, return the first backward drift
        """
        Initialize the IPFP algorithm. Simulate the reference SDE forward and use it to fit the first backward drift.
        Args:
            duration: the duration of the time chunk
            volatility: the volatility of the SDE
            prior_X_0: the prior of the initial location of the particles
        Returns:
            the first backward drift
        """
        # simulate forward the reference SDE. This returns a tensor of time steps and particles spatial-temporal location 
        # (dimension: [num_particles * (num_time_steps), non_temporal_dimension], where the num_particles is set by X0
        # and num_time_steps is set by duration and dt)
        _, Xts = self.sde_solver(b_drift=self.ref_drift, 
                                sigma=volatility, X0=prior_X_0, t0 = t0,dt=self.dt, 
                                N=np.ceil(duration/self.dt).astype(int), 
                                device=self.device)
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1) # flip the time dimension so that we can use this backward
        Ys, Xs = self.make_autoregression(Xts)
        # learn the first backward drift
        self.backward_drift = self.fit(Xs, Ys, volatility)
        gc.collect()
        
    def ipfp_fit_backward_from_forward(self, X_0,duration, volatility, t0 = 0):
        """
        Perform IPFP half bridge. Simulate the SDE forward using the current forward drift, fit the backward drift,
        Args:
            X_0: the initial location of the particles at time 0 (dimension num_particles * non_temporal_dimension)
            X_1: the final location of the particles at time 1 (dimension num_particles * non_temporal_dimension)
            duration: the duration of the time chunk 
            volatility: the volatility of the SDE 
        """
        # simulate forward
        _, Xts = self.sde_solver(b_drift=self.forward_drift, sigma=volatility, 
                              X0=X_0, dt=self.dt, t0 = t0,
                              N=np.ceil(duration/self.dt).astype(int), 
                              device=self.device)
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)
        # fit backward
        Ys, Xs = self.make_autoregression(Xts)
        # learn next step backward drift
        self.backward_drift = self.fit(Xs, Ys, volatility)


    def ipfp_step(self, X_0, X_1,duration, volatility, t0 = 0):
        """
        Perform one step of the IPFP algorithm. Simulate the SDE backward using the current backward drift, fit the forward drift,
        simulate the SDE forward using the current forward drift, fit the backward drift.
        Args:
            X_0: the initial location of the particles at time 0 (dimension num_particles * non_temporal_dimension)
            X_1: the final location of the particles at time 1 (dimension num_particles * non_temporal_dimension)
            duration: the duration of the time chunk 
            volatility: the volatility of the SDE 
        """

        # simulate backward
        _,Xts = self.sde_solver(b_drift=self.backward_drift, 
                            sigma=volatility, X0=X_1, t0 = t0,
                            dt=self.dt, N=np.ceil(duration/self.dt).astype(int), 
                            device=self.device)
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)
        
        # fit forward
        Ys, Xs = self.make_autoregression(Xts)
        self.forward_drift = self.fit(Xs, Ys, volatility)

        # simulate forward
        _, Xts = self.sde_solver(b_drift=self.forward_drift, sigma=volatility, 
                              X0=X_0, dt=self.dt, t0 = t0,
                              N=np.ceil(duration/self.dt).astype(int), 
                              device=self.device)
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)
        # fit backward
        Ys, Xs = self.make_autoregression(Xts)
        # learn next step backward drift
        self.backward_drift = self.fit(Xs, Ys, volatility)

    def ipfp_iteration(self, X_0, X_1,duration, 
                       volatility, t0 = 0, prior_X_0=None,
                       niter = 10 ):
        '''
        solve SBP with two end point using IPFP
        Args:
            X_0: the initial location of the particles at time 0 (dimension num_particles * non_temporal_dimension)
            X_1: the final location of the particles at time 1 (dimension num_particles * non_temporal_dimension)
            duration: the duration of the time chunk 
            volatility: the volatility of the SDE
            prior_X_0: a guess on initial points
            niter: number of iterations
        '''
        if prior_X_0 is None:
            prior_X_0 = X_0
        self.init_ipfp_backward(duration, volatility, t0,
                  prior_X_0)
        for _ in range(niter):
            self.ipfp_step(X_0, X_1,duration, volatility, t0)
        return self.forward_drift



class nndrift(driftfit):
    def __init__(self, model, trainer, N,device = None):
        driftfit.__init__(self, 1./N)
        self.model = model.to(device)
        self.trainer = trainer
        self.device = device
    
    def fit(self, Xs, Ys, volatility):
        Xs = Xs.to(self.device)
        Ys = Ys.to(self.device)
        self.trainer(self.model, Xs, Ys)
        return(self.model.predict)

class generalIPFP(baseIPFP):
    def __init__(self,driftfitting , 
                 ref_drift=None, 
                 sde_solver = solve_sde_RK, 
                 N = 10, device = None):
        self.driftfitting = driftfitting # take an object with make_autoregression and fit method
        baseIPFP.__init__(self, ref_drift, sde_solver, N, device)
    
    def init_ipfp_backward(self, duration, volatility, t0 = 0,
                  prior_X_0=None): #  initialize IPFP, return the first backward drift
        """
        Initialize the IPFP algorithm. Simulate the reference SDE forward and use it to fit the first backward drift.
        Args:
            duration: the duration of the time chunk
            volatility: the volatility of the SDE
            prior_X_0: the prior of the initial location of the particles
        Returns:
            the first backward drift
        """
        # simulate forward the reference SDE. This returns a tensor of time steps and particles spatial-temporal location 
        # (dimension: [num_particles * (num_time_steps), non_temporal_dimension], where the num_particles is set by X0
        # and num_time_steps is set by duration and dt)
        _, Xts = self.sde_solver(b_drift=self.ref_drift, 
                                sigma=volatility, X0=prior_X_0, t0 = t0,dt=self.dt, 
                                N=np.ceil(duration/self.dt).astype(int), 
                                device=self.device)
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1) # flip the time dimension so that we can use this backward
        Ys, Xs = self.driftfitting.make_autoregression(Xts)
        # learn the first backward drift
        self.backward_drift = copy.deepcopy(self.driftfitting.fit(Xs, Ys, volatility))
        for layer in self.driftfitting.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        
        gc.collect()
        
    
    def ipfp_step(self, X_0, X_1,duration, volatility, t0 = 0):
        """
        Perform one step of the IPFP algorithm. Simulate the SDE backward using the current backward drift, fit the forward drift,
        simulate the SDE forward using the current forward drift, fit the backward drift.
        Args:
            X_0: the initial location of the particles at time 0 (dimension num_particles * non_temporal_dimension)
            X_1: the final location of the particles at time 1 (dimension num_particles * non_temporal_dimension)
            duration: the duration of the time chunk 
            volatility: the volatility of the SDE 
        """

        # simulate backward
        _,Xts = self.sde_solver(b_drift=self.backward_drift, 
                            sigma=volatility, X0=X_1, t0 = t0,
                            dt=self.dt, N=np.ceil(duration/self.dt).astype(int), 
                            device=self.device)
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)
        
        # fit forward
        Ys, Xs = self.driftfitting.make_autoregression(Xts)
        #breakpoint()
        self.forward_drift = copy.deepcopy(self.driftfitting.fit(Xs, Ys, volatility))
        for layer in self.driftfitting.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        gc.collect()

        # simulate forward
        _, Xts = self.sde_solver(b_drift=self.forward_drift, sigma=volatility, 
                              X0=X_0, dt=self.dt, t0 = t0,
                              N=np.ceil(duration/self.dt).astype(int), 
                              device=self.device)
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)
        # fit backward
        Ys, Xs = self.driftfitting.make_autoregression(Xts)
        # learn next step backward drift
        self.backward_drift = copy.deepcopy(self.driftfitting.fit(Xs, Ys, volatility))
        for layer in self.driftfitting.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        gc.collect()

    def ipfp_iteration(self, X_0, X_1,duration, 
                       volatility, t0 = 0, prior_X_0=None,
                       niter = 10 ):
        '''
        solve SBP with two end point using IPFP
        Args:
            X_0: the initial location of the particles at time 0 (dimension num_particles * non_temporal_dimension)
            X_1: the final location of the particles at time 1 (dimension num_particles * non_temporal_dimension)
            duration: the duration of the time chunk 
            volatility: the volatility of the SDE
            prior_X_0: a guess on initial points
            niter: number of iterations
        '''
        if prior_X_0 is None:
            prior_X_0 = X_0
        self.init_ipfp_backward(duration, volatility, t0,
                  prior_X_0)
        for _ in range(niter):
            self.ipfp_step(X_0, X_1,duration, volatility, t0)
        return self.forward_drift


class SchrodingerTrain:
    def __init__(self, marginals, dts, volatility):
        """
        Args:
            marginals: the empirical marginals of the data at each time step. List[Tensor], each tensor has dimension [num_particles, non_temporal_dimension
            dts: the times at which we observe the data (Array of length num_time_steps)
            volatility: the volatility of the SDE
        """
        assert len(marginals) == dts.shape[0] 
        self.marginals = marginals
        self.dts = dts
        self.volatility = volatility
        self.N_snapshots = len(marginals)
        self.IPFP = None
        self.fit_trajectory = None
    

    def IPFP_forward_learning(self, IPFPworker, driftworker, iteration = 10, disablebar = False):
        """
        Learn the forward drift using an slightly extended IPFP.
        Args:
            IPFPworker: a callable that generates IPFP object
            iteration: the number of iterations to run IPFP
        Returns:
            drift: learned forward drift
        """
        self.IPFP = [IPFPworker() for _ in range(self.N_snapshots-1) ]
        backward_drift = [None for _ in range(self.N_snapshots-1)]
        Xs = [None for _ in range(self.N_snapshots-1)]
        Ys = [None for _ in range(self.N_snapshots-1)]
        durations = self.dts[1:]-self.dts[:-1]
        for interval in range(self.N_snapshots-1):
            self.IPFP[interval].init_ipfp_backward(durations[interval], 
                                         self.volatility, 
                                         self.dts[interval],
                                         self.marginals[interval])
            backward_drift[interval] = self.IPFP[interval].backward_drift
            #self.IPFP.clear_drifts()
        for _ in tqdm(range(iteration), disable = disablebar):
            # learn forward
            for interval in range(self.N_snapshots-1):
                # simulate from backward
                _,Xts = self.IPFP.sde_solver(b_drift=backward_drift[interval], 
                            sigma=self.volatility, X0=self.marginals[interval+1], t0 = self.dts[interval+1],
                            dt=self.IPFP.dt, N=np.ceil(durations[interval]/self.IPFP.dt).astype(int), 
                            device=self.IPFP.device)
                # Reverse the series
                Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)
        
                # get data for forward fit
                Ys[interval], Xs[interval] = self.IPFP[interval].make_autoregression(Xts)
            Xs_aggr = torch.cat(Xs)
            Ys_aggr = torch.cat(Ys)

            forward_drift =  driftworker.fit(Xs_aggr, Ys_aggr, self.volatility)
            for interval in range(self.N_snapshots-1):
                self.IPFP[interval].forward_drift = forward_drift
                self.IPFP[interval].ipfp_fit_backward_from_forward(self.marginals[interval], durations[interval], self.volatility, self.dts[interval])
                backward_drift[interval] = self.IPFP[interval].backward_drift
                #self.IPFP.clear_drifts()
        return forward_drift



    def IPFPinterpolation(self, IPFPworker, iteration = 10, disablebar = False):
        """
        Learn the interpolated trajectory for each particle in the system.
        Args:
            IPFPworker: a callable making IPFP object
            iteration: the number of iterations to run IPFP
        Returns:
            backnforth: the interpolated trajectories between each marginals (Tensor of dimension [num_particles, num_time_steps, non_temporal_dimension+1])
        """
        #self.IPFP = IPFPworker
        if self.IPFP is None: # if IPFP objects were not made, make them here 
            self.IPFP = [IPFPworker() for _ in range(self.N_snapshots-1) ]
        forward_drift = [None for _ in range(self.N_snapshots-1)]
        backward_drift = [None for _ in range(self.N_snapshots-1)]
        durations = self.dts[1:]-self.dts[:-1]

        # Chop things up and run IPFP to learn forward and backward drifts for each time step 
        for interval in tqdm(range(self.N_snapshots-1), disable = disablebar):
            self.IPFP[interval].init_ipfp_backward(durations[interval], 
                                         self.volatility, 
                                         self.dts[interval],
                                         self.marginals[interval])
            for _ in range(iteration):
                self.IPFP[interval].ipfp_step(self.marginals[interval], 
                                    self.marginals[interval+1],
                                    durations[interval], 
                                    self.volatility, t0 = self.dts[interval])
            forward_drift[interval] = self.IPFP[interval].forward_drift
            backward_drift[interval] = self.IPFP[interval].backward_drift
            # self.IPFP.clear_drifts()
        
        # Generate the trajectories using the learnt forward and backward drifts
        Ns = np.ceil((self.dts[1:]-self.dts[:-1])/self.IPFP[0].dt).astype(int) # number of discretized time steps for each chunk
        backnforth = [None for _ in range(self.N_snapshots)]
        for time_step in range(self.N_snapshots):
            forward_part = None
            backward_part = None
            if time_step < self.N_snapshots-1: # then having forward
                # push things forward one time step 
                _, forward_part = self.IPFP[0].sde_solver(b_drift=forward_drift[time_step], 
                                                              sigma=self.volatility, 
                                                              X0=self.marginals[time_step], 
                                                              t0=self.dts[time_step], dt=self.IPFP[0].dt,
                                                              N=Ns[time_step], device=self.IPFP[0].device) 
                # for as many time steps we have left (going forward), keep pushing forward
                for step in range(time_step+1, self.N_snapshots-1): 
                    _, tmp = self.IPFP[0].sde_solver(b_drift=forward_drift[step], 
                                                        sigma=self.volatility, 
                                                        X0=forward_part[:, -1, :-1], 
                                                        t0 = self.dts[step], 
                                                        dt=self.IPFP[0].dt, 
                                                        N=Ns[step], 
                                                        device=self.IPFP[0].device)
                    forward_part = torch.cat((forward_part, tmp[:, 1:, :]), axis = 1)
            # same thing, just backward
            if time_step > 0: 

                _, backward_part = self.IPFP[0].sde_solver(b_drift=backward_drift[time_step-1], 
                                                              sigma=self.volatility, 
                                                              X0=self.marginals[time_step], 
                                                              t0 = self.dts[time_step-1], # dts[-1] - dts[time_step+1], # backward at ith mid point 
                                                              dt=self.IPFP[0].dt, N=Ns[time_step-1], # solve for interval
                                                              device=self.IPFP[0].device)
                
                for step in range(time_step-1) :
                    _, tmp = solve_sde_RK(b_drift=backward_drift[time_step-2 - step], 
                                                        sigma=self.volatility, X0=backward_part[:, -1, :-1], 
                                                        t0 = self.dts[time_step-2 - step], 
                                                        dt=self.IPFP[0].dt, 
                                                        N=Ns[time_step - 2 - step], 
                                    device=self.IPFP[0].device)
                    backward_part = torch.cat((backward_part, tmp[:, 1:, :]), axis = 1)
                backward_part[:, :, :-1] = backward_part[:, :, :-1].flip(1) # flip backward part
        
            if backward_part is None:
                backnforth[time_step] = forward_part
            elif forward_part is None:
                backnforth[time_step] = backward_part
            else :
                backnforth[time_step] = torch.cat((backward_part[:, :-1, :], 
                                              forward_part), 
                                            axis = 1) # 
             
        backnforth = torch.cat(backnforth, axis = 0)
        return backnforth

    @profile   
    def iter_drift_fit(self, IPFPworker, driftworker, 
                       ipfpiter = 10, 
                       iteration = 10, 
                       disablebar = False):
        '''
        Fitting unknown drift by interatively interpolating and fitting
        Args:
            IPFPworker: an IPFP object
            driftworker: an object to fit drift from trajectories
            ipfpiter: iterations for interpolation 
            iteration: iteration for interpolation and fitting 
            disablebar: whether to disable progress bar
        Returns:
            ref_drift, a callable, fitted drift
            backnforth: the interpolated trajectories between each marginals (Tensor of dimension [num_particles, num_time_steps, non_temporal_dimension+1])
        '''
        self.IPFP = [IPFPworker() for _ in range(self.N_snapshots-1) ]
        for _ in tqdm(range(iteration), disable = disablebar):
            # 
            backnforth = self.IPFPinterpolation(IPFPworker, 
                                                ipfpiter, 
                                                True)
            Ys, Xs = driftworker.make_autoregression(backnforth)
            #breakpoint()
            ref_drift = driftworker.fit(Xs, Ys, self.volatility)
            for i in range(self.N_snapshots-1):
                self.IPFP[i].set_ref_drift(ref_drift)
        return ref_drift, backnforth