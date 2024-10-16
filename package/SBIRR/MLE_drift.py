import pickle
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
from SBIRR.GP import MultitaskGPModel, MultitaskGPModelSparse
from SBIRR.NN import Feedforward, train_nn
from SBIRR.RFF import RandomFourierFeatures
from SBIRR.utils import (auxiliary_plot_routine_init, 
                               auxiliary_plot_routine_end)
from SBIRR.unet import get_trained_unet


def fit_drift_gp(Xts, N, dt, num_data_points=10, num_time_points=50, 
                 kernel=gp.kernels.RBF, noise=1.0, gp_mean_function=None, 
                 nystrom=False, device=None, rff=False, num_rff_features=1000,
                 debug_rff=False, stable=False, nn=False, nn_epochs=100,
                 heteroskedastic=False):
    """
    This function transforms a set of timeseries into an autoregression problem 
    and estimates the drift function using GPs following:
    
        - Papaspiliopoulos, Omiros, Yvo Pokern, Gareth O. Roberts, and 
          Andrew M. Stuart.
          "Nonparametric estimation of diffusions: a differential equations 
           approach."
          Biometrika 99, no. 3 (2012): 511-531.
        - Ruttor, A., Batz, P., & Opper, M. (2013).
          "Approximate Gaussian process inference for the drift function in 
           stochastic differential equations."
          Advances in Neural Information Processing Systems, 26, 2040-2048.
    
    :param Xts[MxNxD ndarray]: Array containing M timeseries of length N of 
        dimension D
    :param N [int]: Number of samples in the time series
    :param dt [float]: time interval seperation between time points (sample 
        rate)
    
    :param num_data_points[int]: Number of inducing samples (inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps (inducing points) 
        for the EM approximation
    :param heteroskedastic: Whether to use heteroskedastic (time-varying)
        noise in the GP. When False, the GP uses the minimum sigma value as 
        before; when True, *and* sigma is a tuple (if sigma is a scalar, this
        argument has no effect), the same time-varying noise is used as in 
        the SDE solver. TODO: make it work for RFF and Nystrom; currently they 
        default to homoskedastic.        
    
    :return [nx(d+1) ndarray-> nxd ndarray]: returns fitted drift
    """
    # Extract starting point
    X_0 = Xts[:, 0, 0].reshape(-1, 1)  

    # Autoregressive targets y = (X_{t+e} - X_t)/dt
    Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1]) / 
          (1 if stable else dt)).reshape((-1, Xts.shape[2] - 1)) 

    # Drop the last timepoint in each timeseries
    Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) 

    if nn:
        return get_trained_unet(Xs, Ys, device=device, num_epochs=nn_epochs, 
                                batch_size=100)
    
    # Set up GP
    elif rff:
        # Default to homoskedastic noise for now
        noise = noise ** 2 if isinstance(noise, (int, float)) else noise[0] ** 2
        rff_model = RandomFourierFeatures(Xs, Ys, num_features=num_rff_features,
                                          kernel=kernel, noise=noise, device=device,
                                          debug_rff=debug_rff)
        return lambda x: rff_model.drift(x) / (dt if stable else 1)
    elif nystrom:
        # Default to homoskedastic noise for now
        noise = noise ** 2 if isinstance(noise, (int, float)) else noise[0] ** 2       
        gp_drift_model = MultitaskGPModelSparse(
            Xs, Ys, dt=1, kern=kernel, noise=noise, 
            gp_mean_function=gp_mean_function, num_data_points=num_data_points, 
            num_time_points=num_time_points, device=device) 

    else:
        if isinstance(noise, (tuple, list)) and heteroskedastic:
            # TODO: make one function for this instead of repeating code in SDE 
            #       solve and here. 
            assert len(noise) == 2
            ti = torch.arange(N).double().to(device) * dt
            sigma_min, sigma_max = noise
            m = 2 * (sigma_max - sigma_min) / (N * dt)  # gradient
            noise = (sigma_max - m * torch.abs(ti - (0.5 * N * dt))).double().to(device)
            noise = noise ** 2  # Need to square noise as we no longer use 
                                # observation_noise (which squares input noise)
            noise = noise.repeat(Xts.shape[0])
        else: 
            noise = noise ** 2 if isinstance(noise, (int, float)) else noise[0] ** 2
                
        gp_drift_model = MultitaskGPModel(Xs, Ys, dt=1 / (dt ** 2 if stable else 1), 
                                          kern=kernel, noise=noise, 
                                          gp_mean_function=gp_mean_function) 
    # fit_gp(gp_drift_model, num_steps=5) # Fit the drift
    
    def gp_ou_drift(x, debug=False):
        return gp_drift_model.predict(x, debug=debug) / (dt if stable else 1)

#     # Extract mean drift
#     gp_ou_drift = lambda x,debug: gp_drift_model.predict(x, debug=debug)  
    return gp_ou_drift


def fit_drift_nn(Xts, N, dt, num_data_points=10, num_time_points=50, 
                 kernel=gp.kernels.RBF, noise=1.0, gp_mean_function=None, 
                 nn_model = Feedforward, device=None):
    """
    This function transforms a set of timeseries into an autoregression problem 
    and estimates the drift function using GPs following:
    
        - Papaspiliopoulos, Omiros, Yvo Pokern, Gareth O. Roberts, and 
          Andrew M. Stuart.
          "Nonparametric estimation of diffusions: a differential equations 
           approach."
          Biometrika 99, no. 3 (2012): 511-531.
        - Ruttor, A., Batz, P., & Opper, M. (2013).
          "Approximate Gaussian process inference for the drift function in 
           stochastic differential equations."
          Advances in Neural Information Processing Systems, 26, 2040-2048.
    
    :param Xts[MxNxD ndarray]: Array containing M timeseries of length N of 
        dimension D
    :param N [int]: Number of samples in the time series
    :param dt [float]: time interval seperation between time points (sample 
        rate)

    :param num_data_points[int]: Number of inducing samples(inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps(inducing points) 
        for the EM approximation
    
    :return [nx(d+1) ndarray-> nxd ndarray]: returns fitted drift
    """
    X_0 = Xts[:, 0, 0].reshape(-1, 1)  # Extract starting point
    
    # Autoregressive targets y = (X_{t+e} - X_t)/dt
    Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1])  ).reshape((-1, Xts.shape[2] - 1)) 
    
    # Drop the last timepoint in each timeseries
    Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) 
    
    n, d = Xs.shape

    nn_drift_model = nn_model(input_size=d).double().to(device) #Setup the NN
    train_nn(nn_drift_model, Xs, Ys)

#     nn_drift_model = LinearRegression()
#     nn_drift_model.fit(Xs, Ys, method="lin")
#     nn_drift_model.eval()
    # fit_gp(gp_drift_model, num_steps=5) # Fit the drift
    
    def gp_ou_drift(x,debug=False):
        return nn_drift_model.predict(x, debug=debug) / dt
#     # Extract mean drift
#     gp_ou_drift = lambda x,debug: gp_drift_model.predict(x, debug=debug)  
    return gp_ou_drift


def MLE_IPFP(
        X_0, X_1, N=10, sigma=1, iteration=10, prior_drift=None,
        num_data_points=10, num_time_points=50, prior_X_0=None, prior_Xts=None,
        num_data_points_prior=None, num_time_points_prior=None, plot=False,
        kernel=gp.kernels.Exponential, observation_noise=1.0, decay_sigma=1, 
        div=1, gp_mean_prior_flag=False, log_dir=None, rff=False,
        verbose=0, langevin=False, nn=False, device=None, nystrom=False,
        num_rff_features=100, debug_rff=False, stable=False, nn_epochs=100,
        log_file_name=None, heteroskedastic=False
    ):
    """
    This module runs the GP drift fit variant of IPFP it takes in samples from 
    \pi_0 and \pi_1 as well as a the forward drift of the prior \P and computes 
    an estimate of the Schroedinger Bridge of \P,\pi_0,\pi_1:
    
                        \Q* = \argmin_{\Q \in D(\pi_0, \pi_1)} KL(\Q || \P)
    
    :params X_0[nxd ndarray]: Source distribution sampled from \pi_0 (initial 
        value for the bridge)
    :params X_1[mxd ndarray]: Target distribution sampled from \pi_1 (terminal 
        value for the bridge)
    
    :param N[int]: number of timesteps for Euler-Maruyama discretisations
    :param iteration[int]: number of IPFP iterations
    
    :param prior_drift[nx(d+1) ndarray-> nxd ndarray]: drift function of the 
        prior, defautls to Brownian

    :param num_data_points[int]: Number of inducing samples(inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps(inducing points) 
        for the EM approximation
    
    :param prior_X_0[mxd array]: The marginal for the prior distribution \P . 
        his is a free parameter which can be tweaked to encourage exploration 
        and improve results.

    :param prior_Xts[nxTxd array] : Prior trajectory that can be used on the 
        first iteration of IPML
    :param num_data_points_prior[int]: number of data inducing points to use for 
        the prior backwards drift estimation prior to the IPFP loop and thus can 
        afford to use more samples here than with `num_data_points`. Note 
        starting off IPFP with a very good estimate of the backwards drift of 
        the prior is very important and thus it is encouraged to be generous 
        with this parameter.
    :param num_time_points_prior[int]: number of time step inducing points to 
        use for the prior backwards drift estimation. Same comments as with 
        `num_data_points_prior`.
    :param decay_sigma[float]: Decay the noise sigma at each iteration.
    :param log_dir[str]: Directory to log the result. If None don't log.
    :param log_file_name[str]: File to log progress on iterations, and other
        useful debugging info as needed.
    
    :return: At the moment returning the fitted forwards and backwards 
        timeseries for plotting. However should also return the forwards and 
        backwards drifts. 
    """

    if log_file_name is not None:
        log = True
        logging.basicConfig(filename=log_file_name, level=logging.INFO, force=True)
        logging.info('Starting logging...')

    if prior_drift is None:
        def prior_drift(x):
            return torch.tensor([0] * (x.shape[1] - 1)).reshape((1, -1)).repeat(x.shape[0], 1).to(device)
        
    # It is easier API wise to have a single function, as otherwise 
    # distinguishing their parameters is fiddly (we can't just say fit_drift = 
    # fit_drift_nn if nn else fit_drift_gp as they take very different params). 
    # Can undo this later, especially if we want to use other types of NN like 
    # Feedforward, but for now this helps with clarity. So fit_drift_nn is never 
    # called. TODO: abstract all the params, so the 3 calls to fit_drift in this 
    # function only differ in the Xts.
    # fit_drift = fit_drift_nn if nn else fit_drift_gp
    fit_drift = fit_drift_gp

    # Setup for the priors backwards drift estimate
    prior_X_0 = X_0 if prior_X_0 is None else prior_X_0        
    num_data_points_prior = (num_data_points if num_data_points_prior is None 
                             else num_data_points_prior)
    num_time_points_prior = (num_time_points if num_time_points_prior is None 
                             else num_time_points_prior)
    drift_forward = None
        
    dt = 1.0 / N
    
    pow_ = int(math.floor(iteration / div))

    # This is now redundant; we do the squaring inside fit_drift_gp (note that
    # we pass in sigma directly instead of observation_noise) and 
    # decay_sigma is not currently supported. 
    # if isinstance(sigma, tuple):
    #     observation_noise = sigma[0] ** 2
    # else:
    #     observation_noise = (sigma ** 2 if decay_sigma == 1.0 
    #                          else (sigma * (decay_sigma ** pow_)) ** 2)


    if langevin:
        d = sigma.shape[0]
        sigma[:int(d * 0.5)] = 0
    
    # Estimating the backward drift of brownian motion
    # Start in prior_X_0 and go forward. 
    # Then flip the series and learn a backward drift: drift_backward
    t, Xts = solve_sde_RK(b_drift=prior_drift, sigma=sigma, X0=prior_X_0, dt=dt, 
                          N=N, device=device)

    T_ = copy.deepcopy(t)
    M_ = copy.deepcopy(Xts)

    if prior_Xts is not None:
        Xts[:, :, :-1] = prior_Xts.flip(1)  # Reverse the series
    else:
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)  # Reverse the series

    drift_backward = fit_drift(
        Xts, N=N, dt=dt, num_data_points=num_data_points_prior,
        num_time_points=num_time_points_prior, kernel=kernel, 
        noise=sigma, gp_mean_function=prior_drift, device=device,
        nystrom=nystrom, rff=rff, num_rff_features=num_rff_features,
        debug_rff=debug_rff, stable=stable, nn=nn, nn_epochs=nn_epochs,
        heteroskedastic=heteroskedastic
    )
    
    if plot and isinstance(sigma, (int, float)):
        auxiliary_plot_routine_init(Xts, t, prior_X_0, X_1, drift_backward, 
                                    sigma, N, dt, device)

    result = []
        
    for i in tqdm(range(iteration)):
        # Estimate the forward drift
        # Start from the end X_1 and then roll until t=0
        if verbose:
            print("Solve drift forward ")
            t0 = time.time()
        t, Xts = solve_sde_RK(b_drift=drift_backward, sigma=sigma, X0=X_1, 
                              dt=dt, N=N, device=device)
        if verbose:
            print("Forward drift solved in ", time.time() - t0)
        del drift_forward
        gc.collect()
        T2 = copy.deepcopy(t.clone().detach())
        M2 = copy.deepcopy(Xts.clone().detach())
        
        if i == 0: result.append([T_, M_, T2, M2])
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)

        if verbose:
            print("Fit drift")
            t0 = time.time()

        del drift_backward
        gc.collect()

        drift_forward = fit_drift(
            Xts, N=N, dt=dt, num_data_points=num_data_points,
            num_time_points=num_time_points, kernel=kernel, 
            noise=sigma, device=device,
            gp_mean_function=(prior_drift if gp_mean_prior_flag else None), 
            nystrom=nystrom, rff=rff, num_rff_features=num_rff_features,
            debug_rff=debug_rff, stable=stable, nn=nn, nn_epochs=nn_epochs,
            heteroskedastic=heteroskedastic
        )
        if verbose:
            print("Fitting drift solved in ", time.time() - t0)

        # Estimate backward drift
        # Start from X_0 and roll until t=1 using drift_forward
        # HERE: HERE is where the GP prior kicks in and helps the most
        t, Xts = solve_sde_RK(b_drift=drift_forward, sigma=sigma, X0=X_0, dt=dt,
                              N=N, device=device)
        
        T = copy.deepcopy(t.clone().detach())
        M = copy.deepcopy(Xts.clone().detach())

        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)

        drift_backward = fit_drift(
            Xts, N=N, dt=dt, num_data_points=num_data_points,
            num_time_points=num_time_points, kernel=kernel, 
            noise=sigma, device=device,
            gp_mean_function=(prior_drift if gp_mean_prior_flag else None), 
            nystrom=nystrom, rff=rff, num_rff_features=num_rff_features,
            debug_rff=debug_rff, stable=stable, nn=nn, nn_epochs=nn_epochs,
            heteroskedastic=heteroskedastic
            # One wouuld think this should (worth rethinking this)
            # be prior drift backwards here
            # but that doesnt work as well,
            # Its kinda clear (intuitively)
            # that prior_drift backwards
            # as a fallback is not going to help
            # this prior, instead the prior of this GP
            # should be inherting the backwards drift
            # of the GP at iteration 1 sadly we dont 
            # have such an estimate thus this should be None
        )

        result.append([T, M, T2, M2])
        if i < iteration and i % div == 0:
            sigma *= decay_sigma

        gc.collect() # fixes odd memory leak
        if log_dir is not None:
            pickle.dump(result, open(f"{log_dir}/result_{i}.pkl" , "wb"))

        logging.info(f"Completed iter {i + 1} of {iteration}")
    
    T2, M2 = solve_sde_RK(b_drift=drift_backward, sigma=sigma, X0=X_1, dt=dt, 
                          N=N, device=device)
    if iteration == 0:
        return [(None, None, T2, M2)]
    
    T, M = solve_sde_RK(b_drift=drift_forward, sigma=sigma, X0=X_0, dt=dt, N=N, 
                        device=device)
    result.append([T, M, T2, M2])
    if log_dir is not None:
        pickle.dump(result, open(f"{log_dir}/result_final.pkl", "wb"))
        
    if plot and isinstance(sigma, (int, float)):
        auxiliary_plot_routine_end(Xts, t, prior_X_0, X_1, drift_backward, 
                                   sigma, N, dt, device)
        
    logging.info("Done.")
    return result


## fit drift with multiple chunck of data
def fit_drift_gp_irr(Xts, Ns, dt, Xts_is_backnforth = True,num_data_points=10, num_time_points=50, 
                 kernel=gp.kernels.RBF, noise=1.0, gp_mean_function=None, 
                 nystrom=False, device=None, rff=False, num_rff_features=1000,
                 debug_rff=False, stable=False, nn=False, nn_epochs=100,
                 heteroskedastic=False):
    """
    This function transforms a set of timeseries into an autoregression problem 
    and estimates the drift function using GPs following:
    
        - Papaspiliopoulos, Omiros, Yvo Pokern, Gareth O. Roberts, and 
          Andrew M. Stuart.
          "Nonparametric estimation of diffusions: a differential equations 
           approach."
          Biometrika 99, no. 3 (2012): 511-531.
        - Ruttor, A., Batz, P., & Opper, M. (2013).
          "Approximate Gaussian process inference for the drift function in 
           stochastic differential equations."
          Advances in Neural Information Processing Systems, 26, 2040-2048.
    
    :param Xts[list of MxNxD ndarray]: Array containing M timeseries of length N of 
        dimension D (including time as the last)
    :param N [int]: Number of samples in the time series
    :param dt [float]: time interval seperation between time points (sample 
        rate)
    
    :param num_data_points[int]: Number of inducing samples (inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps (inducing points) 
        for the EM approximation
    :param heteroskedastic: Whether to use heteroskedastic (time-varying)
        noise in the GP. When False, the GP uses the minimum sigma value as 
        before; when True, *and* sigma is a tuple (if sigma is a scalar, this
        argument has no effect), the same time-varying noise is used as in 
        the SDE solver. TODO: make it work for RFF and Nystrom; currently they 
        default to homoskedastic.        
    
    :return [nx(d+1) ndarray-> nxd ndarray]: returns fitted drift
    """
    

    # Autoregressive targets y = (X_{t+e} - X_t)/dt
    if Xts_is_backnforth:
        Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1]) / 
          (1 if stable else dt)).reshape((-1, Xts.shape[2] - 1)) 
    
        # Drop the last timepoint in each timeseries
        Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) 

    else:
        N_steps = len(Xts)
        Ys = ((Xts[0][:, 1:, :-1] - Xts[0][:, :-1, :-1]) / 
          (1 if stable else dt)).reshape((-1, Xts[0].shape[2] - 1)) 
    
        # Drop the last timepoint in each timeseries
        Xs = Xts[0][:, :-1, :].reshape((-1, Xts[0].shape[2])) 
    
        for i_mid_points in range(N_steps-1):
            Xtmp = Xts[i_mid_points][:, :-1, :].reshape((-1, Xts[i_mid_points].shape[2]))
            Ytmp = ((Xts[i_mid_points][:, 1:, :-1] - Xts[i_mid_points][:, :-1, :-1]) / 
            (1 if stable else dt)).reshape((-1, Xts[i_mid_points].shape[2] - 1))
            Xs = torch.cat((Xs, Xtmp))
            Ys = torch.cat((Ys, Ytmp))

    if nn:
        return get_trained_unet(Xs, Ys, device=device, num_epochs=nn_epochs, 
                                batch_size=100)
    
    # Set up GP
    elif rff:
        # Default to homoskedastic noise for now
        noise = noise ** 2 if isinstance(noise, (int, float)) else noise[0] ** 2
        rff_model = RandomFourierFeatures(Xs, Ys, num_features=num_rff_features,
                                          kernel=kernel, noise=noise, device=device,
                                          debug_rff=debug_rff)
        return lambda x: rff_model.drift(x) / (dt if stable else 1)
    elif nystrom:
        # Default to homoskedastic noise for now
        noise = noise ** 2 if isinstance(noise, (int, float)) else noise[0] ** 2       
        gp_drift_model = MultitaskGPModelSparse(
            Xs, Ys, dt=1, kern=kernel, noise=noise, 
            gp_mean_function=gp_mean_function, num_data_points=num_data_points, 
            num_time_points=num_time_points, device=device) 

    else:
        if isinstance(noise, (tuple, list)) and heteroskedastic:
            # TODO: make one function for this instead of repeating code in SDE 
            #       solve and here. 
            assert len(noise) == 2
            ti = torch.arange(Ns.sum()).double().to(device) * dt
            sigma_min, sigma_max = noise
            m = 2 * (sigma_max - sigma_min) / (Ns.sum() * dt)  # gradient
            noise = (sigma_max - m * torch.abs(ti - (0.5 * Ns.sum() * dt))).double().to(device)
            noise = noise ** 2  # Need to square noise as we no longer use 
                                # observation_noise (which squares input noise)
            noise = noise.repeat(Xts.shape[0])
        else: 
            noise = noise ** 2 if isinstance(noise, (int, float)) else noise[0] ** 2
                
        gp_drift_model = MultitaskGPModel(Xs, Ys, dt=1 / (dt ** 2 if stable else 1), 
                                          kern=kernel, noise=noise, 
                                          gp_mean_function=gp_mean_function) 
    # fit_gp(gp_drift_model, num_steps=5) # Fit the drift
    
    def gp_ou_drift(x, debug=False):
        return gp_drift_model.predict(x, debug=debug) / (dt if stable else 1)

#     # Extract mean drift
#     gp_ou_drift = lambda x,debug: gp_drift_model.predict(x, debug=debug)  
    return gp_ou_drift
    



## routine for irregularly sampled SB chain
def MLE_irrIPFP(
        Xs, dts, N=10, sigma=1, iteration=10, prior_drift=None,
        num_data_points=10, num_time_points=50, prior_X_0=None, prior_Xts=None,
        num_data_points_prior=None, num_time_points_prior=None, plot=False,
        kernel=gp.kernels.Exponential, observation_noise=1.0, decay_sigma=1, 
        div=1, gp_mean_prior_flag=False, log_dir=None, rff=False,
        verbose=0, langevin=False, nn=False, device=None, nystrom=False,
        num_rff_features=100, debug_rff=False, stable=False, nn_epochs=100,
        log_file_name=None, heteroskedastic=False
    ):
    """
    This module runs the GP drift fit variant of IPFP it takes in samples from 
    \pi_t_0 and \pi_t_1...\pi_t_n as well as a the forward drift of the prior \P and computes 
    an estimate of the Schroedinger Bridge of \P,\pi_0,\pi_1:
    
                        \Q* = \argmin_{\Q \in D(\pi_t_0,... \pi_t_n)} KL(\Q || \P)
    
    :params X_s[ list of n_ixd ndarray]: Source distribution sampled from \pi_i (mid values of the bridge chain)
    
    :params dts[len(X_s) x 1 ndarray]: time at marginals
    :param N[int]: number of timesteps for Euler-Maruyama discretisations for one unit time interval
    :param iteration[int]: number of IPFP iterations
    
    :param prior_drift[nx(d+1) ndarray-> nxd ndarray]: drift function of the 
        prior, defautls to Brownian

    :param num_data_points[int]: Number of inducing samples(inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps(inducing points) 
        for the EM approximation
    
    :param prior_X_0[list of n_ixd array]: The marginal for the prior distribution \P . 
        his is a free parameter which can be tweaked to encourage exploration 
        and improve results.

    :param prior_Xts[ list of n_ixTxd array] : Prior trajectory that can be used on the 
        first iteration of IPML
    :param num_data_points_prior[int]: number of data inducing points to use for 
        the prior backwards drift estimation prior to the IPFP loop and thus can 
        afford to use more samples here than with `num_data_points`. Note 
        starting off IPFP with a very good estimate of the backwards drift of 
        the prior is very important and thus it is encouraged to be generous 
        with this parameter.
    :param num_time_points_prior[int]: number of time step inducing points to 
        use for the prior backwards drift estimation. Same comments as with 
        `num_data_points_prior`.
    :param decay_sigma[float]: Decay the noise sigma at each iteration.
    :param log_dir[str]: Directory to log the result. If None don't log.
    :param log_file_name[str]: File to log progress on iterations, and other
        useful debugging info as needed.
    
    :return: At the moment returning the fitted forwards and backwards 
        timeseries for plotting. However should also return the forwards and 
        backwards drifts. 
    """
    N_steps = len(Xs) # number of mid points in the chain
    if log_file_name is not None:
        log = True
        logging.basicConfig(filename=log_file_name, level=logging.INFO, force=True)
        logging.info('Starting logging...')

    if prior_drift is None:
        def prior_drift(x):
            return torch.tensor([0] * (x.shape[1] - 1)).reshape((1, -1)).repeat(x.shape[0], 1).to(device)
        
    # It is easier API wise to have a single function, as otherwise 
    # distinguishing their parameters is fiddly (we can't just say fit_drift = 
    # fit_drift_nn if nn else fit_drift_gp as they take very different params). 
    # Can undo this later, especially if we want to use other types of NN like 
    # Feedforward, but for now this helps with clarity. So fit_drift_nn is never 
    # called. TODO: abstract all the params, so the 3 calls to fit_drift in this 
    # function only differ in the Xts.
    # fit_drift = fit_drift_nn if nn else fit_drift_gp
    fit_drift = fit_drift_gp

    # Setup for the priors backwards drift estimate
    prior_X_0 = Xs if prior_X_0 is None else prior_X_0        
    num_data_points_prior = (num_data_points if num_data_points_prior is None 
                             else num_data_points_prior)
    num_time_points_prior = (num_time_points if num_time_points_prior is None 
                             else num_time_points_prior)
    drift_forward = None
        
    dt = 1.0 / N
    Ns = np.ceil((dts[1:]-dts[:-1])/dt).astype(int) # this calculated time intervals
    
    pow_ = int(math.floor(iteration / div))

    # This is now redundant; we do the squaring inside fit_drift_gp (note that
    # we pass in sigma directly instead of observation_noise) and 
    # decay_sigma is not currently supported. 
    # if isinstance(sigma, tuple):
    #     observation_noise = sigma[0] ** 2
    # else:
    #     observation_noise = (sigma ** 2 if decay_sigma == 1.0 
    #                          else (sigma * (decay_sigma ** pow_)) ** 2)


    if langevin:
        d = sigma.shape[0]
        sigma[:int(d * 0.5)] = 0
    
    # Estimating the backward drift of brownian motion
    # Start in prior_X_0 and go forward. 
    # Then flip the series and learn a backward drift: drift_backward
    t = [None for _ in range(N_steps-1)]
    Xts = [None for _ in range(N_steps-1)] # list of seires, has time as the last coordinates
    drift_backward = [None for _ in range(N_steps-1)]
    T_ = [None for _ in range(N_steps-1)]
    M_ = [None for _ in range(N_steps-1)]

    for i_mid_points in range(N_steps-1):
        t[i_mid_points], Xts[i_mid_points] = solve_sde_RK(b_drift=prior_drift, 
                                                          sigma=sigma, X0=prior_X_0[i_mid_points], dt=dt, 
                                                          N=Ns[i_mid_points], t0 = dts[i_mid_points],device=device)
        if prior_Xts is not None:
            Xts[i_mid_points][:, :, :-1] = prior_Xts[i_mid_points][:, :, :-1].flip(1)  # Reverse the series
            
        else:
            Xts[i_mid_points][:, :, :-1] = Xts[i_mid_points][:, :, :-1].flip(1)  # Reverse the series
        
        drift_backward[i_mid_points] = fit_drift(
            Xts[i_mid_points], N=Ns[i_mid_points], 
            dt=dt, num_data_points=num_data_points_prior,
            num_time_points=num_time_points_prior, kernel=kernel, 
            noise=sigma, gp_mean_function=prior_drift, device=device,
            nystrom=nystrom, rff=rff, num_rff_features=num_rff_features,
            debug_rff=debug_rff, stable=stable, nn=nn, nn_epochs=nn_epochs,
            heteroskedastic=heteroskedastic
        )
        #breakpoint()
        T_[i_mid_points] = copy.deepcopy(t[i_mid_points])
        M_[i_mid_points] = copy.deepcopy(Xts[i_mid_points])

    

    
    
    if plot and isinstance(sigma, (int, float)):
        auxiliary_plot_routine_init(Xts[0], t[0], prior_X_0, Xs[1], drift_backward, 
                                    sigma, Ns[0], dt, device)

    result = [None for _ in range(N_steps-1)]
    
        
    for i in tqdm(range(iteration)):
        # Estimate the forward drift
        # Start from the end X_1 and then roll until t=0
        if verbose:
            print("Solve drifts forward ")
            t0 = time.time()
        
        T2 = [None for _ in range(N_steps-1)]
        M2 = [None for _ in range(N_steps-1)]
        for i_mid_points in range(N_steps-1):
            t[i_mid_points], Xts[i_mid_points] = solve_sde_RK(b_drift=drift_backward[i_mid_points], 
                                                              sigma=sigma, X0=Xs[i_mid_points+1], 
                                                              t0 = dts[i_mid_points], # dts[-1] - dts[i_mid_points+1], # backward at ith mid point 
                                                dt=dt, N=Ns[i_mid_points], # solve for interval
                                                device=device)
            T2[i_mid_points] = copy.deepcopy(t[i_mid_points].clone().detach())
            M2[i_mid_points] = copy.deepcopy(Xts[i_mid_points].clone().detach())

        if verbose:
            print("Forward drifts solved in ", time.time() - t0)
        del drift_forward
        gc.collect()
        
        
        #if i == 0: result.append([T_, M_, T2, M2])
        

        if verbose:
            print("Fit drift")
            t0 = time.time()

        del drift_backward
        gc.collect()

        drift_forward = [None for _ in range(N_steps-1)]
        drift_backward = [None for _ in range(N_steps-1)]
        
        for i_mid_points in range(N_steps-1):
            # Reverse the series
            Xts[i_mid_points][:, :, :-1] = Xts[i_mid_points][:, :, :-1].flip(1)
            drift_forward[i_mid_points] = fit_drift(
                Xts[i_mid_points], N=Ns[i_mid_points], dt=dt, num_data_points=num_data_points,
                num_time_points=num_time_points, kernel=kernel, 
                noise=sigma, device=device,
                gp_mean_function=(prior_drift if gp_mean_prior_flag else None), 
                nystrom=nystrom, rff=rff, num_rff_features=num_rff_features,
                debug_rff=debug_rff, stable=stable, nn=nn, nn_epochs=nn_epochs,
                heteroskedastic=heteroskedastic
            )
        if verbose:
            print("Fitting drift solved in ", time.time() - t0)

        # Estimate backward drift
        # Start from X_0 and roll until t=1 using drift_forward
        # HERE: HERE is where the GP prior kicks in and helps the most
        T = [None for _ in range(N_steps-1)]
        M = [None for _ in range(N_steps-1)]
        for i_mid_points in range(N_steps-1):
            t[i_mid_points], Xts[i_mid_points] = solve_sde_RK(b_drift=drift_forward[i_mid_points], 
                                                              sigma=sigma, X0=Xs[i_mid_points], 
                                                              t0=dts[i_mid_points], dt=dt,
                              N=Ns[i_mid_points], device=device)
        
            T[i_mid_points] = copy.deepcopy(t[i_mid_points].clone().detach())
            M[i_mid_points] = copy.deepcopy(Xts[i_mid_points].clone().detach())

        # Reverse the series
            Xts[i_mid_points][:, :, :-1] = Xts[i_mid_points][:, :, :-1].flip(1)

            drift_backward[i_mid_points] = fit_drift(
                Xts[i_mid_points], N=Ns[i_mid_points], dt=dt, 
                num_data_points=num_data_points,
                num_time_points=num_time_points, kernel=kernel, 
                noise=sigma, device=device,
                gp_mean_function=(prior_drift if gp_mean_prior_flag else None), 
                nystrom=nystrom, rff=rff, num_rff_features=num_rff_features,
                debug_rff=debug_rff, stable=stable, nn=nn, nn_epochs=nn_epochs,
                heteroskedastic=heteroskedastic
            # One wouuld think this should (worth rethinking this)
            # be prior drift backwards here
            # but that doesnt work as well,
            # Its kinda clear (intuitively)
            # that prior_drift backwards
            # as a fallback is not going to help
            # this prior, instead the prior of this GP
            # should be inherting the backwards drift
            # of the GP at iteration 1 sadly we dont 
            # have such an estimate thus this should be None
            )

        #result.append([T, M, T2, M2])
        if i < iteration and i % div == 0:
            sigma *= decay_sigma

        gc.collect() # fixes odd memory leak
        if log_dir is not None:
            pickle.dump(result, open(f"{log_dir}/result_{i}.pkl" , "wb"))

        logging.info(f"Completed iter {i + 1} of {iteration}")
    T2 = [None for _ in range(N_steps-1)]
    M2 = [None for _ in range(N_steps-1)]
    for i_mid_points in range(N_steps-1):
        T2[i_mid_points], M2[i_mid_points] = solve_sde_RK(b_drift=drift_backward[i_mid_points], 
                                                          sigma=sigma, X0=Xs[i_mid_points+1], 
                                                          t0 = dts[i_mid_points], #dts[-1] - dts[i_mid_points+1],
                                                          dt=dt, 
                          N=Ns[i_mid_points], device=device)
    if iteration == 0:
        return [(None, None, T2, M2)]
    
    for i_mid_points in range(N_steps-1):
        T[i_mid_points], M[i_mid_points] = solve_sde_RK(b_drift=drift_forward[i_mid_points], 
                                                        sigma=sigma, X0=Xs[i_mid_points], 
                                                        t0 = dts[i_mid_points], 
                                                        dt=dt, 
                                                        N=Ns[i_mid_points], 
                        device=device)
    result.append([T, M, T2, M2])

    # TODO: for each marginal steps, go back and forward to get entire trajectory 
    backnforth = [None for _ in range(N_steps)] # sample that push to entire time 

    for i_mid_points in range(N_steps):
        forward_part = None
        backward_part = None
        if i_mid_points < N_steps-1: # then having forward
            forward_part = M[i_mid_points] 
            for step in range(i_mid_points+1, N_steps-1): # push things forward
                _, tmp = solve_sde_RK(b_drift=drift_forward[step], 
                                                        sigma=sigma, X0=forward_part[:, -1, :-1], 
                                                        t0 = dts[step], 
                                                        dt=dt, 
                                                        N=Ns[step], 
                        device=device)
                forward_part = torch.cat((forward_part, tmp[:, 1:, :]), axis = 1)
        if i_mid_points > 0: # having backward samples
            backward_part = M2[i_mid_points-1]
            for step in range(i_mid_points-1) :
                _, tmp = solve_sde_RK(b_drift=drift_backward[i_mid_points-2 - step], 
                                                        sigma=sigma, X0=backward_part[:, -1, :-1], 
                                                        t0 = dts[i_mid_points-2 - step], 
                                                        dt=dt, 
                                                        N=Ns[i_mid_points - 2 - step], 
                                    device=device)
                backward_part = torch.cat((backward_part, tmp[:, 1:, :]), axis = 1)
            backward_part[:, :, :-1] = backward_part[:, :, :-1].flip(1) # flip backward part
        
        if backward_part is None:
            backnforth[i_mid_points] = forward_part
        elif forward_part is None:
            backnforth[i_mid_points] = backward_part
        else :
            #breakpoint()
            backnforth[i_mid_points] = torch.cat((backward_part[:, :-1, :], 
                                              forward_part), 
                                            axis = 1) # 
    #breakpoint()
    backnforth = torch.cat(backnforth, axis = 0)
    if log_dir is not None:
        pickle.dump(result, open(f"{log_dir}/result_final.pkl", "wb"))
        
    if plot and isinstance(sigma, (int, float)):
        auxiliary_plot_routine_end(Xts, t, prior_X_0, Xs[1], drift_backward, 
                                   sigma, Ns[0], dt, device)
        
    logging.info("Done.")
    return result, backnforth # return results and a cated trajectories



## iterating between irrIPFP and drift fit

def IPFP_drift(Xs, dts, N=10, sigma=1, iteration=10, IPFP_iter = 10, 
               prior_drift=None, kernel = gp.kernels.Exponential, gp_mean_function = None,
        num_data_points=10, num_time_points=50, prior_X_0=None, prior_Xts=None,
        num_data_points_prior=None, num_time_points_prior=None, plot=False,
        kernelIPFP=gp.kernels.Exponential, gp_mean_functionIPFP = None, observation_noise=1.0, decay_sigma=1, 
        div=1, gp_mean_prior_flag=False, log_dir=None, rff=False,
        verbose=0, langevin=False, nn=False, device=None, nystrom=False,
        num_rff_features=100, debug_rff=False, stable=False, nn_epochs=100,
        log_file_name=None, heteroskedastic=False):
    
    # if no starting point start with BM
    if prior_drift is None:
        def prior_drift(x):
            return torch.tensor([0] * (x.shape[1] - 1)).reshape((1, -1)).repeat(x.shape[0], 1).to(device)
    # start with the prior drift
    drift = prior_drift
    dt = 1/N
    Ns = np.ceil((dts[1:]-dts[:-1])/dt).astype(int)
    for it in range(iteration):
        gc.collect()
        # = MLE_irrIPFP(
        IPFP_res, Xts = MLE_irrIPFP(
            Xs, dts, N=N,sigma=sigma, iteration=IPFP_iter, prior_drift=drift,
            num_data_points=num_data_points, 
            num_time_points=num_time_points, 
            prior_X_0=prior_X_0, 
            prior_Xts = prior_Xts,
            num_data_points_prior=num_data_points_prior, 
            num_time_points_prior=num_time_points_prior, 
            plot=False,
            kernel=kernelIPFP, observation_noise=observation_noise, 
            decay_sigma=decay_sigma, div=div, 
            gp_mean_prior_flag=gp_mean_prior_flag, 
            log_dir=None, rff=rff,
            verbose=verbose, langevin=langevin, 
            nn=nn, device=device, nystrom=nystrom,
            num_rff_features=num_rff_features, 
            debug_rff=debug_rff, stable=stable, 
            nn_epochs=nn_epochs,
            log_file_name=log_file_name, 
            heteroskedastic=heteroskedastic
        )
        if (prior_Xts is None) or (it>0):
            prior_Xts = IPFP_res[-1][1]

        del IPFP_res
        gc.collect()
        #breakpoint()
        
        drift = fit_drift_gp_irr(Xts, Ns, dt, Xts_is_backnforth =True,
                                 num_data_points=num_data_points, 
                                 num_time_points=num_time_points, 
                                 kernel=kernel, noise=sigma, 
                                 gp_mean_function=gp_mean_function, 
                                 nystrom=nystrom, device=device, 
                                 rff=rff, num_rff_features=num_rff_features,
                                 debug_rff=debug_rff, stable=stable, 
                                 nn=nn, nn_epochs=nn_epochs,
                                 heteroskedastic=heteroskedastic)
        
    
    if log_dir is not None:
        NotImplemented
        #pickle.dump(drift, open(f"{log_dir}/drift_final.pkl", "wb"))
    return(drift)
    #raise(NotImplementedError)
