import torch
import torch.nn as nn

# NOTE: The shape of the input data is (n, 2) where n is the number of data points.
# The shape of the output data MUST BE (n, 1) for the loss function to work properly (and not just tensor of size n)

# Class for neural network to learn the potential function from observation. 
# We will differentiate the potential function to get the gradient field.

class GradFieldMLP(nn.Module):
    
    def __init__(self, input_size=3, hidden_size=100, droptime = True):
        '''
        Input_size should be spatial dimension+1, if droptime = True the last dimension is dropped
        '''
        super(GradFieldMLP, self).__init__()
        input_size = input_size - droptime
        self.input_size = input_size
        self.hidden_size  = hidden_size
        # Define your architecture here
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, 1)  
        self.droptime = droptime

    def forward(self, x):
        if self.droptime:
            x = x[:,:-1] # drop last dimension that is time
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        x.requires_grad_(True)
        potential = self.forward(x)
        grad_field = torch.autograd.grad(potential, x, grad_outputs=torch.ones_like(potential), create_graph=True)[0]
        grad_field = grad_field[:, :-1] # drop gradient for time. 
        return grad_field 

class GradFieldMLP2(nn.Module):
    def __init__(self, net, droptime = True):
        '''
        Input_size should be spatial dimension+1, if droptime = True the last dimension is dropped
        '''
        super(GradFieldMLP2, self).__init__()
        self.net = net
        self.droptime = droptime

    def forward(self, x):
        if self.droptime:
            x = x[:,:-1] # drop last dimension that is time
        return self.net(x)
    
    def predict(self, x):
        x.requires_grad_(True)
        potential = self.forward(x)
        grad_field = torch.autograd.grad(potential, x, grad_outputs=torch.ones_like(potential), create_graph=True)[0]
        grad_field = grad_field[:, :-1] # drop gradient for time. 
        return grad_field 

class timedepMLP(nn.Module):
    
    def __init__(self, input_size=3, hidden_size=100):
        '''
        Input_size should be spatial dimension+1, if droptime = True the last dimension is dropped
        '''
        super(timedepMLP, self).__init__()
        input_size = input_size 
        self.input_size = input_size
        self.hidden_size  = hidden_size
        # Define your architecture here
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, input_size - 1)  

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        out = self.fc2(hidden)
        return out
    
    def predict(self, x):
        return self.forward(x) 




def train_nn(model, x_train, y_train, epochs = 20, lr=0.05, verbose=False):
    """
    Train the neural network model to learn the potential function.
    Args:
        model: The neural network model.
        x_train: The input data for training.
        y_train: The output data for training (the potential function)
    Returns:
        The trained neural network model.
    """

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = loss_function(y_pred, y_train)
        if verbose:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
    return model

def train_nn_gradient(model, x_train, y_train_gradient, optimizer = None,
                      epochs = 20, lr = 0.05,verbose=False):
    """
    Train the neural network model to learn the gradient field.
    Args:
        model: The neural network model.
        x_train: The input data for training.
        y_train_gradient: The output data for training (the gradient field)
        epochs: The number of epochs to train the model.
        lr: The learning rate.
        verbose: Whether to print the loss at each epoch.
    Returns:
        The trained neural network model.
    """

    loss_function = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict(x_train)
        # Compute Loss
        loss = loss_function(y_pred, y_train_gradient)
        if verbose:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
    return model