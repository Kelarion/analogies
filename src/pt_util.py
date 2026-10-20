import os, sys
import pickle

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import numpy as np
import scipy
import scipy.linalg as la
import scipy.special as spc
import numpy.linalg as nla
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
import itertools as itt

def batch_data(*data, **dl_args):
	""" 
	wrapper for weird data batching in pytorch
	data supplied as positional arguments, must have same size in first dimension 
	"""
	dset = torch.utils.data.TensorDataset(*data)
	return torch.utils.data.DataLoader(dset, **dl_args)

##### transformer utils
def sinusoidal_encoding(seqs):

    p = torch.arange(seqs.shape[0])[:,None,None]*torch.ones(1,1,seqs.shape[-1])
    d = torch.arange(seqs.shape[-1])[None,None,:]*torch.ones(seqs.shape[0],1,1)

    args = p/(1000**(2*(d//2)/seqs.shape[-1]))

    return torch.where(np.mod(d,2)>0, np.cos(args), np.sin(args))

##### weight initializations
def uniform_tensor(*dims):
	""" returns a tensor for random uniform values between -1 and 1 """

	return torch.FloatTensor(*dims).uniform_(-1,1)

def sphere_weights(*dims):
	w = torch.FloatTensor(*dims).normal_()
	return w / w.norm(dim=0, keepdim=True)


def train_model(mod, dl, epochs=100, verbose=False, **opt_args):

    ls = []
    for it in range(epochs):
        ls.append(mod.grad_step(dl, **opt_args))

    return ls 

#####################################################################
################ data loading from torchvision ######################
#####################################################################

def cifar(train=True):

    transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                        download=True, transform=transform)

    cond = torch.tensor(dset.targets)
    inps = torch.tensor(dset.data.transpose(0,3,1,2)).float()

    return inps, cond

def mnist(train=True):

    dset = torchvision.datasets.MNIST(root='./data', train=train,
                                        download=True)

    cond = dset.targets
    inps = dset.data 

    return inps, cond

#####################################################################

class BalancedBinary:

	def __init__(self, *Ks, normalize=False):
		""" 
		include all signed K-combinations of output weights, with 
		K>0 dimensions being set to +/- 1

		for example, if there are 3 output dimensions, K=3 would be 
		[+,+,+] [-,+,+] [+,-,+], [-,-,+], [+,+,-], etc.

		while K=1 would be 
		[+,0,0],[-,0,0],[0,+,0], etc.

		In general, there are (N-choose-K)*(2^K) unique output weights
		for each choice of K. They'll be divided evenly among the inputs
		"""

		self.Ks = Ks
		self.normalize = normalize

		self.__name__ = f"balanced_binary_{Ks}"

	def __call__(self, *dims):
		"""
		BalancedBinary(dim_out, dim_in)
		"""
        
		dim_out = dims[-2]
		dim_in = dims[-1]

		if self.Ks is None:
			self.Ks = [dim_out]

		# number of unique weight vectors per group
		tot_per_grp = [spc.binom(dim_out, k)*(2**k) for k in self.Ks]

		num_per_grp = int(dim_in//np.sum(tot_per_grp)) # number of neurons per group
		rmnd = int(np.mod(dim_in, np.sum(tot_per_grp)))
		extras = np.random.choice(int(sum(tot_per_grp)), rmnd, replace=False) # random groups get an extra

		# print(rmnd)
		# print(num_per_grp)

		W = []
		for k in self.Ks:
			# all +/- labels of the k conditions
			bin_vals = 2*(np.mod(np.arange(2**k)[:,None]//(2**np.arange(k)[None,:]), 2))-1 
			
			# print(bin_vals.shape)
			# label the chosen conditions accordingly
			for c in combinations(range(dim_out),k):
				w_grp = np.zeros((dim_out,(2**k)))
				w_grp[c,:] = bin_vals.T

				W.append(w_grp)

		Ws = np.concatenate(W, axis=1)
		reps = np.array([int(num_per_grp) + 1*(i in extras) for i in range(int(sum(tot_per_grp)))])
		Ws = np.repeat(Ws, reps, axis=1)

		if self.normalize:
			Ws /= la.norm(Ws, axis=0, keepdims=True)

		return Ws

# class 

# def balanced_binary(dim_out, dim_in):




#     bits_1 = np.concatenate([np.eye(dim_out)*i for i in [1,-1]])
# 	bits_2 = 2*(np.mod(np.arange(2**dim_out)[:,None]//(2**np.arange(dim_out)[None,:]),2)) - 1

#     num_pop = len(bits_1)
#     num_per_pop = dim_in//num_pop

#     which_pop = np.arange(num_per_pop*num_pop)//num_per_pop
#     leftovers = np.random.choice(num_pop, dim_in - num_per_pop*num_pop, replace=False)
#     if shuffle:
#         which_pop = np.random.permutation(np.append(which_pop, leftovers))
#     else:
#         which_pop = np.append(which_pop, leftovers)

##### Custom activation functions
class RayLou(nn.ReLU):
    def __init__(self, linear_grad=False):
        super(RayLou,self).__init__()
        self.linear_grad = linear_grad

        self.__name__ = self.__class__.__name__

    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (x>0).float()

    def __repr__(self):
        return f"RayLou"
    
    # def __repr__(self):
    #     return f"RayLou({'linear' if self.linear_grad else ''})"

class RayLou6(nn.ReLU6):
    def __init__(self, linear_grad=False):
        super(RayLou6,self).__init__()
        self.linear_grad = linear_grad

    def __repr__(self):
        return f"RayLou6({'linear' if self.linear_grad else ''})"

    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return ((x<6)&(x>0)).float()

class RayLouUB(nn.ReLU6):
    def __init__(self, ub=1, linear_grad=False):
        super(RayLouUB,self).__init__()
        self.linear_grad = linear_grad
        self.ub = ub

        # self.__class__.__name__ = f"RayLou{ub:.1f}"
        self.__name__ = f"RayLou{ub:.1f}"

    def forward(self, x):
        return nn.ReLU6()(x*6/self.ub)/6*self.ub

    def __repr__(self):
        return f"RayLouUB({self.ub:.1f})"

    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return ((x<=self.ub)&(x>0)).float()

class RayLouHinge(nn.ReLU):
    def __init__(self, slope=1, linear_grad=False):
        super(RayLouUB,self).__init__()
        self.linear_grad = linear_grad
        self.slope = slope

        # self.__class__.__name__ = f"RayLou{ub:.1f}"
        self.__name__ = f"RayLouHinge{slope:.1f}"

    def forward(self, x):
        return nn.ReLU()(x*6/self.ub)

    def __repr__(self):
        return f"RayLouHinge({self.ub:.1f})"

    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return ((x<=self.ub)&(x>0)).float()

class RayLouShift(nn.ReLU):
    def __init__(self, shift=0, linear_grad=False):
        super(RayLouShift,self).__init__()
        self.linear_grad = linear_grad
        self.shift = shift

        # self.__class__.__name__ = f"RayLou{ub:.1f}"
        self.__name__ = f"RayLou{shift:.1f}"

    def forward(self, x):
        return nn.ReLU()(x + self.shift)

    def __repr__(self):
        return f"RayLouShift({self.shift:.1f})"


    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return ((x>-self.shift)).float()

class LeakyRayLou(nn.LeakyReLU):
    def __init__(self, eps=-1e-2, linear_grad=False):
        super(LeakyRayLou,self).__init__(negative_slope=eps)
        self.linear_grad = linear_grad
        self.eps = eps
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (self.eps*(x<0) + 1*(x>0)).float()

class Poftslus(nn.Softplus):
    def __init__(self, beta=1, linear_grad=False):
        super(Poftslus,self).__init__(beta=beta)
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (1/(1+torch.exp(-self.beta*x))).float()

class NoisyRayLou(nn.ReLU):
    def __init__(self, beta=1, linear_grad=False):
        super(NoisyRayLou,self).__init__()
        self.linear_grad = linear_grad
        self.beta = beta
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return 0.5*(1+torch.erf(x/(self.beta*np.sqrt(2)))).float()

class TanAytch(nn.Tanh):
    def __init__(self, linear_grad=False, rand_grad=False):
        super(TanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.__name__ = self.__class__.__name__
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return 1-nn.Tanh()(x).pow(2)

    # def __repr__(self):
    #     return f"TanAytch({'linear' if self.linear_grad else ''})"
    def __repr__(self):
        return f"TanAytch"


class NoisyTanAytch(nn.Tanh):
    def __init__(self, noise=1, linear_grad=False, rand_grad=False):
        super(NoisyTanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.noise = noise
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return torch.exp(-x.pow(2)/(1+(2*self.noise**2)))

class HardTanAytch(nn.Hardtanh):
    def __init__(self, linear_grad=False, rand_grad=False, vmin=-1, vmax=1):
        super(HardTanAytch,self).__init__(vmin, vmax)
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.vmin = vmin
        self.vmax = vmax
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return ((x<self.vmax)&(x>self.vmin)).float()

class Iden(nn.Identity):
    def __init__(self, linear_grad=False):
        super(Iden,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return torch.ones(x.shape)

#######