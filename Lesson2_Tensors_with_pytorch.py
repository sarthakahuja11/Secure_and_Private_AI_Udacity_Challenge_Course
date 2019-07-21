# First, import PyTorch
import torch

def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))


### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

## Calculate the output of this network using the weights and bias tensors
y1 = torch.sum(features * weights) + bias
#OR: y1 = activation((features * weights).sum() + bias)
y=activation(y1)

## Calculate the output of this network using matrix multiplication
y1 = torch.mm(features,weights.view(5,1)) + bias
y=activation(y1)

## Your solution here

y1 = activation(torch.mm(features,W1)+B1)
y2 = activation(torch.mm(y1,W2)+B2)
print(y2)





----------------

import numpy as np

a = np.random.rand(4,3)
a

b = torch.from_numpy(a)
b

b.numpy()

# Multiply PyTorch Tensor by 2, in place
b.mul_(2)


#The memory is shared between the Numpy array and Torch tensor, so if you change
#the values in-place of one object, the other will change as well



# Numpy array matches new values from Tensor
a








