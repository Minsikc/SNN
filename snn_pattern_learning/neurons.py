import torch 
import torch.nn as nn
import math

class Boxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, subthresh):
        # spike threshold, Heaviside
        # store membrane potential before reset

        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # surrogate-gradient, BoxCar
        # stored membrane potential before reset
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        #print(grad_input)
        temp = abs(input - ctx.thresh) < ctx.subthresh
        #print(temp)
        # return grad_input, None, None
        return grad_input * temp.float(), None, None
    
    
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        #temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma

class HeavisideBoxcarCall(nn.Module):
    def __init__(self, thresh=0.4, subthresh=0.1, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        self.thresh = torch.tensor(thresh)
        self.subthresh = torch.tensor(subthresh)
        self.thresh.to("cuda" if torch.cuda.is_available() else "cpu")
        self.subthresh.to("cuda" if torch.cuda.is_available() else "cpu")
        if spiking:
            self.f = Boxcar.apply
        else:
            self.f = self.primitive_function

    def forward(self, x, thresh=None):
        if thresh is not None:
            return self.f(x, thresh, self.subthresh)
        return self.f(x, self.thresh, self.subthresh)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha)
    
class TriangleCall(nn.Module):
    def __init__(self, thresh=0.6, subthresh=0.25, alpha=1.0, spiking=True, gamma=0.3, width=1):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        self.thresh = torch.tensor(thresh)
        self.subthresh = torch.tensor(subthresh)
        self.gamma = gamma
        self.width = width
        self.thresh.to("cuda" if torch.cuda.is_available() else "cpu")
        self.subthresh.to("cuda" if torch.cuda.is_available() else "cpu")
        if spiking:
            self.f = Triangle.apply

    def forward(self, x, thresh):
        return self.f(x, thresh, self.gamma,self.width)            

class Triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, gamma, width):
        # spike threshold, Heaviside
        # store membrane potential before reset

        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.gamma = gamma
        ctx.width = width
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # surrogate-gradient, Triangle
        # stored membrane potential before reset
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        # temp is now based on the provided equation
        temp = ctx.gamma * torch.max(torch.zeros_like(input), 1 - torch.abs((input - ctx.thresh) / ctx.width*ctx.thresh))
        
        # return grad_input multiplied by temp, None, None, None (since no gradient is calculated for thresh, subthresh, and gamma)
        return grad_input * temp.float(), None, None, None

    
class ATan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, thresh):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone() 
        grad_x = 1 / (1 + (math.pi * input_tensor).square()) * grad_input
        # print("check", input_tensor)    
        # print("grad_x", grad_x)   
        return grad_x, None, None


class ATanCall:
    def __init__(self, thresh=1.0, subthresh=0.5):
        self.alpha = 2.0
        self.thresh=thresh
        

    def __call__(self, x):
        return ATan.apply(x, self.alpha, self.thresh)


class LIF_Node(nn.Module):
    def __init__(self, surrogate_function=TriangleCall(), initial_thresh=0.5):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.thresh = initial_thresh  # Threshold 값 추가

    def forward(self, mem: torch.Tensor, spike_before: torch.Tensor, decay: torch.Tensor, I_in: torch.Tensor):
        mem = mem * decay * (1 - spike_before) + I_in
        spike = self.surrogate_function(mem, self.thresh)  # thresh 인자를 추가로 전달, triangular function을 사용할 경우 추가로 전달
        #spike = self.surrogate_function(mem)  # thresh 인자를 추가로 전달
        return mem, spike
    
class PLIF_Node(nn.Module):
    def __init__(self, surrogate_function=HeavisideBoxcarCall(), initial_thresh=0.6, initial_tau=0.65):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.thresh = nn.Parameter(torch.tensor(initial_thresh))  # Threshold as a learnable parameter
        self.tau = nn.Parameter(torch.tensor(initial_tau))        # Tau (decay rate) as a learnable parameter

    def forward(self, mem: torch.Tensor, spike_before: torch.Tensor, I_in: torch.Tensor):
        # Use learnable tau (decay rate) in the forward pass
        decay = self.tau
        mem = mem * (1 - decay) * (1 - spike_before) + I_in 
        spike = self.surrogate_function(mem, self.thresh)  # Pass the learnable threshold to the surrogate function
        return mem, spike

class ALIF_Node(nn.Module):
    def __init__(self, surrogate_function=ATanCall()):
        super().__init__()
        self.surrogate_function = surrogate_function

    def forward(self, mem: torch.Tensor, spike_before: torch.Tensor, decay: torch.Tensor, I_in: torch.Tensor):
        mem = mem * decay *(1-spike_before)+ I_in
        spike = self.surrogate_function(mem)
        return mem, spike



class Accumulate_node(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mem: torch.Tensor, decay: torch.Tensor, I_in: torch.Tensor):
        mem = mem*(decay)+ I_in*(1-decay)
        return mem
    


b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


