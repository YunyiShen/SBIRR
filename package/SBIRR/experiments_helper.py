import torch
import torch.nn as nn
import numpy as np

import pyro.contrib.gp as gp
from SBIRR.SDE_solver import solve_sde_RK
from SBIRR.utils import plot_trajectories_2
import matplotlib.pyplot as plt
from SBIRR.Schrodinger import gpIPFP, SchrodingerTrain, gpdrift, nndrift
from SBIRR.GP import spatMultitaskGPModel
from SBIRR.gradient_field_NN import train_nn_gradient, GradFieldMLP2
import torch
import torch.nn as nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LotkaVolterra(nn.Module):
    def __init__(self, alpha=0., beta=0., gamma=0., delta=0.):
        super(LotkaVolterra, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.delta = nn.Parameter(torch.tensor(delta))
        self.relu = torch.abs

    def forward(self, x):
        alpha = self.relu(self.alpha) # things needs to be positive
        beta = self.relu(self.beta)
        gamma = self.relu(self.gamma)
        delta = self.relu(self.delta)
        dxdt =  alpha * x[:,0] - beta * x[:,0] * x[:,1]
        dydt = delta * x[:,0] * x[:,1] - gamma * x[:,1]
        return torch.stack([dxdt, dydt], dim = 1)
    def predict(self, x):
        return self.forward(x)
    
## get a shine_ecoli model
class repressilator(nn.Module):
    def __init__(self, beta = 1., n = 1., k = 1., gamma = 1.):
        super(repressilator, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.n = nn.Parameter(torch.tensor(n))
        self.k = nn.Parameter(torch.tensor(k))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.relu = torch.abs

    def forward(self, x):
        
        beta = self.relu(self.beta)
        n = self.relu(self.n)
        k = self.relu(self.k)
        gamma = self.relu(self.gamma)
        x = torch.relu(x) + 1e-8 # concentration has to be positive
        dxdt = beta/(1.+ (x[:,2]/k) ** n) - gamma * x[:,0]
        dydt = beta/(1.+ (x[:,0]/k) ** n) - gamma * x[:,1]
        dzdt = beta/(1.+ (x[:,1]/k) ** n) - gamma * x[:,2]
        return torch.stack([dxdt, dydt, dzdt], dim = 1)
    def predict(self, x):
        return self.forward(x)
    
class spring_model(nn.Module):
    def __init__(self, dim = 2):
        super(spring_model, self).__init__()
        self.dim = dim
        self.hooke = nn.Parameter(torch.zeros(1)+.1) #nn.Linear(dim, dim, bias = False)
        #self.tt = nn.Parameter(torch.tensor(time_scale))
        self.process = torch.abs
    
    def forward(self, x):
        k = self.process(self.hooke)
        dxdt = -x[:,:-1] * k # velocity follow a Hook law
        return dxdt
    def predict(self, x):
        return self.forward(x)
    
class lamboseen(nn.Module):
    def __init__(self):
        super(lamboseen, self).__init__()
        self.x0 = nn.Parameter(torch.zeros(1))
        self.y0 = nn.Parameter(torch.zeros(1))
        self.logscale = nn.Parameter(torch.tensor(0.))
        self.circulation = nn.Parameter(torch.tensor(0.))
    
    def forward(self, x):
        xx = (x[:,0] - self.x0)* torch.exp(-self.logscale)
        y = (x[:,1] - self.y0) * torch.exp(-self.logscale)
        r = torch.sqrt(xx ** 2 + y ** 2)
        theta = torch.atan2(y, xx)
        dthetadt = 1./r * (1- torch.exp(-r**2))
        dxdt = -dthetadt * torch.sin(theta)
        dydt = dthetadt * torch.cos(theta)
        return self.circulation * torch.stack([dxdt, dydt], dim = 1)
    def predict(self, x):
        return self.forward(x)

class twin_lamboseen(nn.Module):
    def __init__(self):
        super(twin_lamboseen, self).__init__()
        self.lamb1 = lamboseen()
        self.lamb2 = lamboseen()
    def forward(self, x):
        return self.lamb1(x) + self.lamb2(x)
    def predict(self, x):
        return self.forward(x)


class big_vortex(nn.Module):
    def __init__(self):
        super(big_vortex, self).__init__()
        self.x0 = nn.Parameter(torch.zeros(1))
        self.y0 = nn.Parameter(torch.zeros(1))
        self.logyscale = nn.Parameter(torch.tensor(0.))
        self.scale = nn.Parameter(torch.tensor(0.))
        

        #self.tt = nn.Parameter(torch.tensor(time_scale))
        self.process = torch.exp
    
    def forward(self, x):
        xx = x[:,0] - self.x0
        y = (x[:,1] - self.y0) * torch.exp(-self.logyscale)

        dxdt = y
        dydt = -xx
        return self.scale * torch.stack([dxdt, dydt], dim = 1)

    def predict(self, x):
        return self.forward(x)
    


class two_vortecs(nn.Module):
    def __init__(self):
        super(big_vortex, self).__init__()
        self.x01 = nn.Parameter(torch.tensor(-0.5))
        self.y01 = nn.Parameter(torch.zeros(1))
        self.x02 = nn.Parameter(torch.tensor(0.5))
        self.y02 = nn.Parameter(torch.zeros(1))
        self.logyscale1 = nn.Parameter(torch.tensor(0.))
        self.scale1 = nn.Parameter(torch.tensor(0.))
        self.logyscale2 = nn.Parameter(torch.tensor(0.))
        self.scale2 = nn.Parameter(torch.tensor(0.))

        #self.tt = nn.Parameter(torch.tensor(time_scale))
        self.process = torch.exp
    
    def forward(self, x):
        xx1 = x[:,0] - self.x01
        y1 = (x[:,1] - self.y01) * torch.exp(-self.logyscale1)
        xx2 = x[:,0] - self.x02
        y2 = (x[:,1] - self.y02) * torch.exp(-self.logyscale2)

        dxdt1 = y1
        dydt1 = -xx1
        dxdt2 = y2
        dydt2 = -xx2
        return self.scale1 * torch.stack([dxdt1, dydt1], dim = 1) + self.scale2 * torch.stack([dxdt2, dydt2], dim = 1)

    def predict(self, x):
        return self.forward(x)    


def get_settings(taskname, N_steps):
    if "GoMvortex" in taskname:
        sigma = 0.1
        dt = 0.02 
        N = int(math.ceil(1.0/dt))
        dts = 2 * np.array([i for i in range(N_steps)])
        ours = big_vortex()
        oursnndrift = nndrift(ours.double().to(device), train_nn_gradient, N = N)
        vanilla = big_vortex()
        vanillanndrift = nndrift(vanilla.double().to(device), train_nn_gradient, N = N)
        return sigma, dt, N, dts, oursnndrift, vanillanndrift
    
    if "LV" in taskname:
        sigma = 0.1
        dt = 0.02 
        N = int(math.ceil(1.0/dt))
        dts = 2 * np.array([i for i in range(N_steps)])
        ours = LotkaVolterra(1e-5, 1e-5, 1e-5, 1e-5)
        oursnndrift = nndrift(ours.double().to(device), train_nn_gradient, N = N)
        vanilla = LotkaVolterra()
        vanillanndrift = nndrift(vanilla.double().to(device), train_nn_gradient, N = N)
        return sigma, dt, N, dts, oursnndrift, vanillanndrift
        
    
    if "repres" in taskname:
        
        sigma = 0.1
        dt = 0.01 #0.005
        N = int(math.ceil(1.0/dt))
        dts = 1.5 * np.array([i for i in range(N_steps)])
        ours = repressilator(1.e-5, 1., 1., 1.e-5)
        oursnndrift = nndrift(ours.double().to(device), train_nn_gradient, N = N)
        vanilla = repressilator(0.,1.,1.,0.)
        vanillanndrift = nndrift(vanilla.double().to(device), train_nn_gradient, N = N)
        return sigma, dt, N, dts, oursnndrift, vanillanndrift

    if "eb" in taskname:
        # SDE Solver config
        sigma = 0.1
        dt = 0.05
        N = int(math.ceil(1.0/dt))
        dts = np.array([i for i in range(N_steps)]) 
        mynet = nn.Sequential(nn.Linear(5 , 128), 
                      nn.ReLU(), 
                      nn.Linear(128,64), 
                      nn.ReLU(), 
                      nn.Linear(64,64), 
                      nn.ReLU(), 
                      nn.Linear(64, 1))
        ours = GradFieldMLP2(net = mynet)
        def train_nn_gradient_verbos(model, x_train, y_train_gradient):
            train_nn_gradient(model, x_train, y_train_gradient, optimizer = None,
                      epochs = 20, lr = 0.01,verbose=False)
        oursnndrift = nndrift(ours.double().to(device), train_nn_gradient_verbos, N = N)
        mynet_vSB = nn.Sequential(nn.Linear(5 , 1, bias = False))
        nn.init.constant_(mynet_vSB[0].weight, 0)
        vanilla = GradFieldMLP2(net = mynet_vSB) 
        vanillanndrift = nndrift(vanilla.double().to(device), train_nn_gradient, N = N)
        return sigma, dt, N, dts, oursnndrift, vanillanndrift

    if "cmu" in taskname:
        sigma = .1 # .05??
        dt = 0.02
        N = int(math.ceil(1.0/dt))
        dts = np.array([i for i in range(N_steps)])  # should make irregular eventually 
        ours = spring_model().double()
        ours.to(device)
        oursnndrift = nndrift(ours, train_nn_gradient, N = N)
        mynet_vSB = nn.Sequential(nn.Linear(2 , 1, bias = False))
        nn.init.constant_(mynet_vSB[0].weight, 0)
        vanilla = GradFieldMLP2(net = mynet_vSB) 
        vanillanndrift = nndrift(vanilla.double().to(device), train_nn_gradient, N = N)
        return sigma, dt, N, dts, oursnndrift, vanillanndrift
    
    if "hESC" in taskname:
        dts = np.array([0, 1, 3])
        sigma = .05
        dt = 0.025
        N = int(math.ceil(1.0/dt))
        mynet = nn.Sequential(nn.Linear(5 , 128), 
                      nn.ReLU(), 
                      nn.Linear(128,64), 
                      nn.ReLU(), 
                      nn.Linear(64,64), 
                      nn.ReLU(), 
                      nn.Linear(64, 1))
        ours = GradFieldMLP2(net = mynet)
        def train_nn_gradient_verbos(model, x_train, y_train_gradient):
            train_nn_gradient(model, x_train, y_train_gradient, optimizer = None,
                      epochs = 30, lr = 0.002,verbose=False)
        oursnndrift = nndrift(ours.double().to(device), train_nn_gradient_verbos, N = N)
        mynet_vSB = nn.Sequential(nn.Linear(5 , 1, bias = False))
        nn.init.constant_(mynet_vSB[0].weight, 0)
        vanilla = GradFieldMLP2(net = mynet_vSB) 
        vanillanndrift = nndrift(vanilla.double().to(device), train_nn_gradient, N = N)
        return sigma, dt, N, dts, oursnndrift, vanillanndrift

