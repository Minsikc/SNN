import torch
import torch.nn as nn
from neurons import LIF_Node, PLIF_Node, HeavisideBoxcarCall, Accumulate_node, TriangleCall
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import logging

logger = logging.getLogger(__name__)

try:
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice
    from aihwkit.simulator.tiles import AnalogTile
except ImportError:
    AnalogLinear = None
    SingleRPUConfig = None
    ConstantStepDevice = None
    AnalogTile = None

class Basic_RSNN_BRP(nn.Module):
    def __init__(self, n_in=100, n_hidden=200, n_out=20, subthresh=0.5, init_tau=2.0, init_spk_trace_tau=0.5):
        super(Basic_RSNN_BRP, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out  # 이제 사용하지 않지만, 네트워크 설계를 위해 남겨둠
        self.subthresh = subthresh
        self.init_tau = init_tau

        self.fc1 = nn.Parameter(0.05 * torch.rand(self.n_in, self.n_hidden) / np.sqrt(self.n_in))
        self.recurrent = nn.Parameter(0.05 * torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden))
        
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall())
    
    def forward(self, x):
        device = x.device
        outputs = []
        num_steps = x.size(1)
        batch_size = x.size(0)
        x = x.view(batch_size, num_steps, -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=device)
        
        for step in range(num_steps):
            input_spike = x[:, step, :]
            hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau, 
                                                 torch.mm(input_spike, self.fc1) + torch.mm(hidden_spike, self.recurrent))
            
            # 순환 뉴런의 1/10만을 출력으로 사용
            outputs.append(hidden_spike[:, :self.n_hidden // 10])
        
        return torch.stack(outputs, dim=1), num_steps

class Basic_RSNN_mem(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        init_tau: float = 0.9,              # membrane decaying time constant
        init_spk_trace_tau: float = 0.5,    # spike trace decaying time constant
        recurrent = False
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        with torch.no_grad():
            self.recurrent.fill_diagonal_(0)
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.ac_node = Accumulate_node()
        self.recurrent_mode=recurrent

    
    def forward(self, x):
        self.device = x.device
        # x.shape = [batch_size, time, channel, width, height]

        outputs = [] 
        
        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device = self.device)
        

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_mode:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+0.1*torch.mm(hidden_spike, self.recurrent))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike))

            out_mem = self.ac_node(out_mem, self.init_tau, self.out(hidden_spike))
            outputs.append(out_mem)

        return torch.stack(outputs, dim=1), num_steps
        # return next - softmax and cross-entropy loss

class Basic_RSNN_spike(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = False,
        init_tau: float = 0.60,              # membrane decaying time constant
        weight_scale: float = 0.5,           # initial weight scaling factor
    ):
        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.weight_scale = weight_scale
        self.recurrent_connection = recurrent
        self.custom_grad = False
        self.custom_grad_forward = False
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        zero_ratio=0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= self.weight_scale

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= self.weight_scale
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        self.LIF0 = LIF_Node(surrogate_function=TriangleCall())
        self.out_node = LIF_Node(surrogate_function=TriangleCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        #self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        torch.nn.init.kaiming_normal_(self.recurrent)
    
    def forward(self, x):
        self.device = x.device
        # x.shape = [batch_size, time, channel, width, height]

        outputs = [] 
        
        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device = self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        #sparse_effective_recurrent=effective_recurrent*self.binary_tensor.to(self.device)
        

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike, self.recurrent))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))
            outputs.append(out_spike)

        return torch.stack(outputs, dim=1)

class Basic_RSNN_spike2(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = False,
        init_tau: float = 0.60,              # membrane decaying time constant    # spike trace decaying time constant
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.recurrent_connection = recurrent
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        zero_ratio=0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden+self.n_out, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden+self.n_out, self.n_hidden+self.n_out)/np.sqrt(self.n_hidden))

        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        self.LIF0 = LIF_Node(surrogate_function=TriangleCall())

        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        #self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        torch.nn.init.kaiming_normal_(self.recurrent)
    
    def forward(self, x):
        self.device = x.device
        # x.shape = [batch_size, time, channel, width, height]

        outputs = [] 
        
        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size,self.n_hidden+self.n_out, device=self.device)

        #sparse_effective_recurrent=effective_recurrent*self.binary_tensor.to(self.device)
        

        for step in range(num_steps):
            input_spike = x[:, step,:]

            hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike, self.recurrent))

            outputs.append(hidden_spike[:, -self.n_out:])

        return torch.stack(outputs, dim=1)

class RSNN_with_delay_synapse(nn.Module):
    def __init__(
        self,
        n_in=100,
        n_hidden=200,
        n_out=20,
        subthresh=0.5,
        init_tau: float = 0.96,
        max_delay: int = 5  # Maximum synaptic delay
    ):
        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.max_delay = max_delay

        zero_ratio = 0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.out_node = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        torch.nn.init.kaiming_uniform_(self.recurrent, a=np.sqrt(5))
        # Initialize the delay mask
        self.delay_mask = torch.zeros(self.n_hidden, self.n_hidden, self.max_delay)
        for i in range(n_hidden):
            for j in range(n_hidden):
                delay = torch.randint(0, max_delay, (1,)).item()
                self.delay_mask[i, j, delay] = 1

    def forward(self, x):
        self.device = x.device

        outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        sparse_effective_recurrent = effective_recurrent * self.binary_tensor.to(self.device)

        # Buffer to store spikes for delay processing
        spike_buffer = torch.zeros(batch_size, self.max_delay, self.n_hidden, device=self.device)

        for step in range(num_steps):
            input_spike = x[:, step, :]

            # Update spike buffer
            spike_buffer = torch.roll(spike_buffer, shifts=-1, dims=1)
            spike_buffer[:, -1, :] = hidden_spike

            # Initialize recurrent input with delays
            recurrent_input = torch.zeros(batch_size, self.n_hidden, device=self.device)

            # Sum the contributions of the delayed spikes
            for t in range(self.max_delay):
                delayed_spikes = spike_buffer[:, t, :]
                delayed_synapse = self.recurrent * self.delay_mask[:, :, t].to(self.device)
                recurrent_input += torch.mm(delayed_spikes, delayed_synapse)

            # Update hidden states with delayed spikes
            hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,
                                                 self.fc1(input_spike) + recurrent_input)
            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))
            outputs.append(out_spike)

        return torch.stack(outputs, dim=1)

class RSNN_singlelayer(nn.Module):
    def __init__(
        self,
        n_neuron = 200,
        subthresh = 0.5,
        init_tau: float = 0.96,              # membrane decaying time constant  
    ):
        super().__init__()

        self.n_neuron = n_neuron
        self.recurrent = nn.Parameter(torch.rand(self.n_neuron, self.n_neuron)/np.sqrt(self.n_neuron))
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        self.LIF = LIF_Node(surrogate_function=HeavisideBoxcarCall())


    def forward(self, x):
        self.device = x.device
        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        
        outputs = []
        for step in range(num_steps):
            input_current = x[:, step,:]
            hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,input_current+0.1*torch.mm(hidden_spike, effective_recurrent))
            outputs.append(hidden_spike)

        return torch.stack(outputs, dim=1)
    


class RSNN_merged_hidden_output(nn.Module):
    def __init__(self, n_in=100, n_hidden=200, n_out=20, subthresh=0.5, init_tau=0.96):
        super().__init__()
        self.n_in = n_in
        self.n_neurons = n_hidden+n_out  # Total neurons including hidden and output
        self.n_hidden = n_hidden  # Just to keep track, not actually used separately
        self.n_out = n_out  # Just to keep track, not actually used separately
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.custom_grad=False
        self.custom_grad_forward = False
        
        # Combining hidden and output neurons into one layer
        self.fc = nn.Linear(self.n_in, self.n_neurons, bias=False)
        for param in self.fc.parameters():
            param.data *= 3  # Initialize weights to small values
            param.requires_grad = False
        self.recurrent = nn.Parameter(torch.rand(self.n_neurons, self.n_neurons)/np.sqrt(self.n_neurons))
        self.LIF = LIF_Node(surrogate_function=TriangleCall())  # Using one LIF node for all neurons
        
        # Mask to zero out self-connections and control sparsity
        self.mask = torch.ones(self.n_neurons, self.n_neurons) - torch.eye(self.n_neurons)
        zero_ratio = 0.3
        random_tensor = torch.rand(self.n_neurons, self.n_neurons)
        self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        
    def forward(self, x):
        self.device = x.device
        num_steps = x.size(1)
        batch_size = x.size(0)
        
        mem = spike = torch.zeros(batch_size, self.n_neurons, device=self.device)
        outputs = []
        
        # Applying combined weight masks for sparsity and avoiding self-connections
        effective_recurrent = self.recurrent * self.mask.to(self.device) * self.binary_tensor.to(self.device)
        
        for step in range(num_steps):
            input_spike = x[:, step, :]
            mem, spike = self.LIF(mem, spike, self.init_tau, self.fc(input_spike) + torch.mm(spike, effective_recurrent))
            outputs.append(spike[:, -self.n_out:])  # Only capturing the last 20 outputs which act as output neurons

        return torch.stack(outputs, dim=1)



class RSNN_fixed_w_in(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = False,
        init_tau: float = 0.60,              # membrane decaying time constant    # spike trace decaying time constant
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.recurrent_connection = recurrent
        self.custom_grad = False
        self.custom_grad_forward = False
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        zero_ratio=0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        for param in self.fc1.parameters():
            param.data *= 3  # Initialize weights to small values
            param.requires_grad = False

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        self.LIF0 = LIF_Node(surrogate_function=TriangleCall())
        self.out_node = LIF_Node(surrogate_function=TriangleCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        #self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        torch.nn.init.kaiming_normal_(self.recurrent)
    
    def forward(self, x):
        self.device = x.device
        # x.shape = [batch_size, time, channel, width, height]

        outputs = [] 
        
        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device = self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        #sparse_effective_recurrent=effective_recurrent*self.binary_tensor.to(self.device)
        

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike, self.recurrent))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))
            outputs.append(out_spike)

        return torch.stack(outputs, dim=1)

# Additional necessary class definitions for LIF_Node, etc., need to be defined accordingly.


# Define Gaussian function
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

class Basic_RSNN_ALIF(nn.Module):
    def __init__(
        self,
        n_in=100,
        n_hidden=200,
        n_out=20,
        subthresh=0.5,
        recurrent=True,
        init_tau: float = 0.96,  # membrane decaying time constant
        tau_range: float = 0.02,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.recurrent_connection = recurrent
        self.init_tau = init_tau
        self.tau_range = tau_range

        zero_ratio = 0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(self.n_hidden, self.n_hidden)
        self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        self.tau_hidden = nn.Parameter((init_tau - tau_range) + 2 * tau_range * torch.rand(self.n_hidden))
        self.tau_out = nn.Parameter((init_tau - tau_range) + 2 * tau_range * torch.rand(self.n_hidden))
        # Define constants for adaptive LIF neuron
        self.b_j0 = 0.01
        self.tau_m = torch.tensor(20)
        self.R_m = torch.tensor(1) 
        self.dt = 1
        self.gamma = .5
        self.lens = 0.5

        torch.nn.init.kaiming_uniform_(self.recurrent, a=np.sqrt(5))
    
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = .15
        lens = 0.5
        temp = torch.exp(-(input**2) / (2 * lens**2)) / torch.sqrt(2 * torch.tensor(math.pi)) / lens
        temp = temp * (1. + hight) - temp * hight * torch.exp(-input / scale) - temp * hight * torch.exp(input / scale)
        return grad_input * temp.float() * 0.5

def mem_update_adp(self, inputs, mem, spike, tau_adp, tau_m, b, isAdapt=1):
    alpha = torch.exp(-1./ tau_m).cuda()
    ro = torch.exp(-1./ tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = self.b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * self.R_m * inputs - B * spike 
    inputs_ = mem - B
    spike = self.ActFun_adp.apply(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b

def forward(self, x):
    self.device = x.device

    outputs = []

    num_steps = x.size(1)
    batch_size = x.size(0)

    hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
    out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)
    effective_recurrent = self.recurrent * self.mask.to(self.device)
    sparse_effective_recurrent = effective_recurrent * self.binary_tensor.to(self.device)
    
    b = torch.zeros(batch_size, self.n_hidden, device=self.device)  # adaptive threshold

    for step in range(num_steps):
        input_spike = x[:, step, :]
        if self.recurrent_connection:
            hidden_mem, hidden_spike, _, b = self.mem_update_adp(self.fc1(input_spike) + 0.7 * torch.mm(hidden_spike, sparse_effective_recurrent),
                                                                    hidden_mem, hidden_spike, self.tau_hidden, self.tau_m, b)
        else:
            hidden_mem, hidden_spike, _, b = self.mem_update_adp(self.fc1(input_spike),
                                                                    hidden_mem, hidden_spike, self.tau_hidden, self.tau_m, b)

        out_mem, out_spike, _, _ = self.mem_update_adp(self.out(hidden_spike), out_mem, out_spike, self.tau_out, self.tau_m, b)
        outputs.append(out_spike)

    return torch.stack(outputs, dim=1)

class Basic_RSNN_STDP(nn.Module):
    def __init__(
        self,
        n_in=100,
        n_hidden=200,
        n_out=20,
        subthresh=0.5,
        recurrent=True,
        init_tau: float = 0.96,  # membrane decaying time constant
        tau_range: float = 0.02,
        stdp_time_window: int = 3  # Time window for STDP updates
    ):
        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.recurrent = recurrent
        self.stdp_time_window = stdp_time_window

        zero_ratio = 0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.out_node = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        self.tau_hidden = nn.Parameter((init_tau - tau_range) + 2 * tau_range * torch.rand(self.n_hidden))

        # Initialize spike history for STDP
        self.spike_history = torch.zeros(self.n_hidden, stdp_time_window)

    def stdp_update(self, spike_window, current_spike):
        # STDP window parameters
        pos_update = 0.01  # positive update magnitude
        neg_update = -0.01  # negative update magnitude

        # Calculate the outer product differences
        for t in range(self.stdp_time_window):
            time_diff = spike_window[:, t].unsqueeze(1) - current_spike.unsqueeze(0)
            weight_update = torch.zeros_like(self.recurrent.data)
            weight_update += (time_diff <= 0).float() * pos_update
            weight_update += (time_diff > 0).float() * neg_update
            self.recurrent.data += weight_update

    def forward(self, x):
        self.device = x.device

        outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        sparse_effective_recurrent = effective_recurrent * self.binary_tensor.to(self.device)

        for step in range(num_steps):
            input_spike = x[:, step, :]
            if self.recurrent:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.tau_hidden, self.fc1(input_spike) + 0.1 * torch.mm(hidden_spike, sparse_effective_recurrent)
                )
            else:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.tau_hidden, self.fc1(input_spike)
                )

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))
            outputs.append(out_spike)

            # Update spike history
            self.spike_history = torch.roll(self.spike_history, shifts=-1, dims=1)
            self.spike_history[:, -1] = hidden_spike.squeeze()

            # STDP weight update for recurrent connections
            if self.recurrent:
                self.stdp_update(self.spike_history, hidden_spike)

        return torch.stack(outputs, dim=1)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

class Basic_RSNN_spike_PLIF(nn.Module):
    def __init__(
        self,
        n_in=100,
        n_hidden=200,
        n_out=20,
        subthresh=0.5,
        recurrent=True,
        init_tau: float = 0.60,
        init_thresh=0.6  # Spike threshold
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.recurrent_connection = recurrent

        # Weight initialization
        zero_ratio = 0.3
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5

        # PLIF nodes with learnable threshold and tau
        self.LIF0 = PLIF_Node(surrogate_function=TriangleCall(), initial_thresh=init_thresh, initial_tau=init_tau)
        self.out_node = PLIF_Node(surrogate_function=TriangleCall(), initial_thresh=init_thresh, initial_tau=init_tau)

        # Mask for recurrent connections
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        torch.nn.init.kaiming_normal_(self.recurrent)
    
    def forward(self, x):
        self.device = x.device

        outputs = []
        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        
        for step in range(num_steps):
            input_spike = x[:, step, :]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, I_in=self.fc1(input_spike) + torch.mm(hidden_spike, self.recurrent))
            else:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, I_in=self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, I_in=self.out(hidden_spike))
            outputs.append(out_spike)

        return torch.stack(outputs, dim=1)


class RSNN_eprop(nn.Module):
    
    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, model, classif, w_init_gain, lr_layer, t_crop, visualize, visualize_light, device):    
        
        super(RSNN_eprop, self).__init__()
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_t      = n_t
        self.thr      = thr
        self.dt       = dt
        self.alpha    = np.exp(-dt/tau_m)
        self.kappa    = np.exp(-dt/tau_o)
        self.gamma    = gamma
        self.b_o      = b_o
        self.model    = model
        self.classif  = classif
        self.lr_layer = lr_layer
        self.t_crop   = t_crop  
        self.visu     = visualize
        self.visu_l   = visualize_light
        self.device   = device
        
        #Parameters
        self.w_in  = nn.Parameter(torch.Tensor(n_rec, n_in ))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.reg_term = torch.zeros(self.n_rec).to(self.device)
        self.B_out = torch.Tensor(n_out, n_rec).to(self.device)
        self.reset_parameters(w_init_gain)

    def reset_parameters(self, gain):
        
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0]*self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1]*self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2]*self.w_out.data
        
    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        
        #Hidden state
        self.v  = torch.zeros(n_t,n_b,n_rec).to(self.device)
        self.vo = torch.zeros(n_t,n_b,n_out).to(self.device)
        #Visible state
        self.z  = torch.zeros(n_t,n_b,n_rec).to(self.device)
        self.zo = torch.zeros(n_t,n_b,n_out).to(self.device)
        #Weight gradients
        self.w_in.grad  = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)
    
    def forward(self, x, yt, do_training):
        
        self.n_b = x.shape[1]    # Extracting batch size
        self.init_net(self.n_b, self.n_t, self.n_in, self.n_rec, self.n_out)    # Network reset
        
        identity = torch.eye(self.n_rec, self.n_rec, device=self.device)
        self.w_rec.data *= (1 - identity)    # Making sure recurrent self excitation/inhibition is cancelled
        
        for t in range(self.n_t-1):     # Computing the network state and outputs for the whole sample duration
        
            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output, vo: output layer membrane potential (yo incl. activation function)
            self.v[t+1]  = (self.alpha * self.v[t] + torch.mm(self.z[t], self.w_rec.t()) + torch.mm(x[t], self.w_in.t())) - self.z[t]*self.thr
            self.z[t+1]  = (self.v[t+1] > self.thr).float()
            self.vo[t+1] = self.kappa * self.vo[t] + torch.mm(self.z[t+1], self.w_out.t())
            self.zo[t+1] = (self.vo[t+1] > self.thr).float()
        
        if self.classif:        #Apply a softmax function for classification problems
            yo = F.softmax(self.vo,dim=2)
        else:
            yo = self.zo

        if do_training:
            self.compute_gradients(x.permute(1, 0, 2), yo.permute(1, 0, 2), yt.permute(1, 0, 2))
            
        return yo
    
    def compute_gradients(self, x, output, target):
        # Surrogate derivative of spike function
        h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - self.thr) / self.thr))

        # Eligibility traces for input and recurrent weights
        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1, -1).to(self.device)

        # Input eligibility trace (from input spikes)
        # x.shape: [batch, step, neuron] -> permute to [batch, neuron, step]
        trace_in = F.conv1d(x.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t, groups=self.n_in)[:, :, 1:self.n_t+1].unsqueeze(1)
        # Apply the surrogate derivative (h)
        trace_in = torch.einsum('btr,brit->brit', h, trace_in)

        # Recurrent eligibility trace (from recurrent spikes)
        trace_rec = F.conv1d(self.z.permute(0, 2, 1), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t, groups=self.n_rec)[:, :, 1:self.n_t+1].unsqueeze(1)
        # Apply the surrogate derivative (h)
        trace_rec = torch.einsum('btr,brit->brit', h, trace_rec)

        # Output eligibility trace (filtered spikes from recurrent neurons)
        kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1, -1).to(self.device)
        trace_out = F.conv1d(self.z.permute(0, 2, 1), kappa_conv.expand(self.n_rec, -1, -1), padding=self.n_t, groups=self.n_rec)[:, :, 1:self.n_t+1]

        # Compute the error signal (assumed to be Cross-Entropy loss derivative)
        err = output - target  # [batch, steps, neurons]

        # Apply convolution kernel to smooth error signal
        kernel_size = 10  # Example value
        decay_rate = 2.0  # Example value
        kernel = self.create_exponential_kernel(kernel_size, decay_rate).to(self.device)
        err = self.apply_convolution(err, kernel, kernel_size)

        # Learning signals (L)
        L = torch.einsum('bto,or->brt', err, self.w_out)

        # Weight gradient updates
        # Input weights (W_in) - using trace_in
        self.w_in.grad += self.lr_layer[0] * torch.sum(L.unsqueeze(2).expand(-1, -1, self.n_in, -1) * trace_in, dim=(0, 3))

        # Recurrent weights (W_rec) - using trace_rec
        self.w_rec.grad += self.lr_layer[1] * torch.sum(L.unsqueeze(2).expand(-1, -1, self.n_rec, -1) * trace_rec, dim=(0, 3))

        # Output weights (W_out) - using trace_out
        self.w_out.grad += self.lr_layer[2] * torch.einsum('bto,brt->or', err, trace_out)


    def create_exponential_kernel(self, kernel_size, decay_rate):
        t = torch.arange(kernel_size, dtype=torch.float32)
        kernel = torch.exp(-torch.flip(t, [0]) / decay_rate)
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel_size)
    
    def apply_convolution(self, spike_train, kernel, kernel_size):
        kernel = kernel.expand(spike_train.size(2), 1, kernel_size)
        spike_train_permuted = spike_train.permute(1, 2, 0)

        # 패딩을 적용하여 길이 방향으로 1D 컨볼루션을 적용
        padded_spike_train = F.pad(spike_train_permuted, (kernel_size - 1, 0), 'constant', 0)
        smoothed_spike_train = F.conv1d(padded_spike_train, kernel, groups=spike_train.size(2))
        
        # 결과 텐서를 원래 차원 순서로 변환 [길이, 배치 크기, 뉴런 수]
        smoothed_spike_train_out = smoothed_spike_train.permute(2, 0, 1)
        return smoothed_spike_train_out
        
    def __repr__(self):
        
        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '
        
class Basic_RSNN_eprop_minsik(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = True,
        init_tau: float = 0.60,
        init_thresh = 0.6,  # Spike threshold
        init_tau_o = 0.6,
        gamma = 0.3,              # membrane decaying time constant    # spike trace decaying time constant
        width = 1
        
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = False
 
        self.gamma = gamma
        self.width = width
        self.thr = init_thresh
        self.tau_o = init_tau_o
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        #self.fc1.weight.requires_grad = False

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        # init_thresh reaches the spiking nodes, not only self.thr (which is
        # used for the surrogate derivative). Without this the student always
        # fired at LIF_Node's default 0.5 regardless of init_thresh, so a
        # teacher built at any other threshold was a DIFFERENT network and its
        # weights were not a solution: planting them into the student gave
        # loss 0.848 instead of 0. With this wired in -- and the teacher built
        # from the same init_thresh/init_tau -- the teacher's own weights are
        # an exact optimum (loss 0.000000, verified for thresh 0.1..0.5).
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall(),
                             initial_thresh=init_thresh)
        self.out_node = LIF_Node(surrogate_function=HeavisideBoxcarCall(),
                                 initial_thresh=init_thresh)
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        torch.nn.init.kaiming_normal_(self.recurrent)

    def init_net(self):
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)


    def forward(self, x):
        self.device = x.device
        self.init_net()                                       
        # x.shape = [batch_size, time, channel, width, height]
        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device = self.device)
        #effective_recurrent = self.recurrent * self.mask.to(self.device)
        #sparse_effective_recurrent=effective_recurrent*self.binary_tensor.to(self.device)
        

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike, self.recurrent))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))
            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)

        return torch.stack(self.outputs, dim=1)


    def compute_grads(self, x, err):
        #x.shape = [batch_size, time, n_in]
        #err.shape = [time, batch, n_out]
        self.n_t = x.size(1)
        self.n_b = x.size(0)
        self.alpha = self.init_tau
        self.v = torch.stack(self.hidden_mem_list, dim=1).permute(1, 0, 2)
        self.z = torch.stack(self.hidden_spike_list, dim=1).permute(1, 0, 2)
        self.vo = torch.stack(self.outputs, dim=1).permute(1, 0, 2)

        # Surrogate derivatives
        h = self.init_tau*torch.max(torch.zeros_like(self.v), 1-torch.abs((self.v-self.thr)/self.thr))
   
        alpha_conv  = torch.tensor([self.alpha ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_in    = F.conv1d(     x.permute(0,2,1), alpha_conv.expand(self.n_in ,-1,-1), padding=self.n_t, groups=self.n_in )[:,:,1:self.n_t+1].unsqueeze(1).expand(-1,self.n_hidden,-1,-1)  #n_b, n_rec, n_in , n_t 
        trace_in    = torch.einsum('tbr,brit->brit', h, trace_in )                                                                                                                          #n_b, n_rec, n_in , n_t 
        trace_rec   = F.conv1d(self.z.permute(1,2,0), alpha_conv.expand(self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_hidden)[:,:, :self.n_t  ].unsqueeze(1).expand(-1,self.n_hidden,-1,-1)  #n_b, n_rec, n_rec, n_t
        trace_rec   = torch.einsum('tbr,brit->brit', h, trace_rec)                                                                                                                          #n_b, n_rec, n_rec, n_t    
        trace_reg   = trace_rec

        # Output eligibility vector (vectorized computation, model-dependent)
        kappa_conv = torch.tensor([self.tau_o ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_out  = F.conv1d(self.z.permute(1,2,0), kappa_conv.expand(self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_hidden)[:,:,1:self.n_t+1]  #n_b, n_rec, n_t

        # Eligibility traces
        trace_in     = F.conv1d(   trace_in.reshape(self.n_b,self.n_in *self.n_hidden,self.n_t), kappa_conv.expand(self.n_in *self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_in *self.n_hidden)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_hidden,self.n_in ,self.n_t)   #n_b, n_rec, n_in , n_t  
        trace_rec    = F.conv1d(  trace_rec.reshape(self.n_b,self.n_hidden*self.n_hidden,self.n_t), kappa_conv.expand(self.n_hidden*self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_hidden*self.n_hidden)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_hidden,self.n_hidden,self.n_t)   #n_b, n_rec, n_rec, n_t
        
        L = torch.einsum('tbo,or->brt', err, self.out.weight)
        
        # Weight gradient updates
        self.fc1.weight.grad  += 0.05*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * trace_in , dim=(0,3)) 
        self.recurrent.grad += 0.05*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_hidden,-1) * trace_rec, dim=(0,3))
        self.out.weight.grad += 0.05*torch.einsum('tbo,brt->or', err, trace_out)

    def pseudo_derivative(self, v_membrane):
        # Pseudo-derivative for spike generation (non-differentiability)
        # You can define it similar to the one in the paper (e.g., soft threshold function)
        return torch.max(torch.zeros_like(v_membrane), 1 - torch.abs(v_membrane - self.thr) / self.thr)
    
    def reset_parameters(self):
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        #self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5

class Basic_RSNN_eprop_minsik_(nn.Module):
    def __init__(
        self,
        n_in=100,
        n_hidden=200,
        n_out=20,
        subthresh=0.5,
        recurrent=True,
        init_tau: float = 0.60,
        init_thresh=0.6,  # Spike threshold
        init_tau_o=0.3,
        width=1
    ):
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = False
        self.width = width
        self.thr = init_thresh
        self.tau_o = init_tau_o
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.out_node = LIF_Node(surrogate_function=HeavisideBoxcarCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        torch.nn.init.kaiming_normal_(self.recurrent)

    def init_net(self):
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)

    def forward(self, x):
        self.device = x.device
        self.init_net()
        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.out_mem_list = []  ## ADDED: 출력 뉴런의 막전위를 저장할 리스트
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)

        for step in range(num_steps):
            input_spike = x[:, step, :]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau, self.fc1(input_spike) + torch.mm(hidden_spike, self.recurrent))
            else:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau, self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.tau_o, self.out(hidden_spike))
            
            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)
            self.out_mem_list.append(out_mem)  ## ADDED: 각 타임스텝의 출력 뉴런 막전위 저장

        return torch.stack(self.outputs, dim=1)

    def compute_grads(self, x, err):
        # x.shape = [batch_size, time, n_in]
        # err.shape = [time, batch, n_out]
        self.n_t = x.size(1)
        self.n_b = x.size(0)
        self.alpha = self.init_tau
        self.gamma = self.tau_o
        self.v = torch.stack(self.hidden_mem_list, dim=1).permute(1, 0, 2)
        self.z = torch.stack(self.hidden_spike_list, dim=1).permute(1, 0, 2)
        
        ## ADDED: 저장된 출력 뉴런의 막전위를 불러옵니다.
        self.vo_mem = torch.stack(self.out_mem_list, dim=1).permute(1, 0, 2)

        # Surrogate derivatives
        # 은닉층 뉴런의 유사-미분
        h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - self.thr) / self.thr))
        
        ## ADDED: 출력층 뉴런의 유사-미분 계산
        # 출력 뉴런도 같은 임계값(thr)을 사용한다고 가정합니다. 다르다면 별도의 파라미터로 관리해야 합니다.
        ho = self.gamma * torch.max(torch.zeros_like(self.vo_mem), 1 - torch.abs((self.vo_mem - self.thr) / self.thr))

        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1, -1).to(self.device)
        trace_in = F.conv1d(x.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t, groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_hidden, -1, -1)
        trace_in = torch.einsum('tbr,brit->brit', h, trace_in)
        trace_rec = F.conv1d(self.z.permute(1, 2, 0), alpha_conv.expand(self.n_hidden, -1, -1), padding=self.n_t, groups=self.n_hidden)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_hidden, -1, -1)
        trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)

        # Output eligibility vector
        kappa_conv = torch.tensor([self.tau_o ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1, -1).to(self.device)
        trace_out = F.conv1d(self.z.permute(1, 2, 0), kappa_conv.expand(self.n_hidden, -1, -1), padding=self.n_t, groups=self.n_hidden)[:, :, 1:self.n_t + 1]

        # Eligibility traces
        trace_in = F.conv1d(trace_in.reshape(self.n_b, self.n_in * self.n_hidden, self.n_t), kappa_conv.expand(self.n_in * self.n_hidden, -1, -1), padding=self.n_t, groups=self.n_in * self.n_hidden)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_hidden, self.n_in, self.n_t)
        trace_rec = F.conv1d(trace_rec.reshape(self.n_b, self.n_hidden * self.n_hidden, self.n_t), kappa_conv.expand(self.n_hidden * self.n_hidden, -1, -1), padding=self.n_t, groups=self.n_hidden * self.n_hidden)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_hidden, self.n_hidden, self.n_t)

        ## CHANGED: 오차 신호(err)에 출력 뉴런의 유사-미분(ho)을 곱하여 학습 신호(L)를 계산합니다.
        modulated_err = err * ho
        # tbo, or -> tbr 후 -> brt 로 변환
        L = torch.einsum('tbo,or->tbr', modulated_err, self.out.weight).permute(1, 2, 0)

        # Weight gradient updates
        self.fc1.weight.grad += 0.05 * torch.sum(L.unsqueeze(2).expand(-1, -1, self.n_in, -1) * trace_in, dim=(0, 3))
        self.recurrent.grad += 0.05 * torch.sum(L.unsqueeze(2).expand(-1, -1, self.n_hidden, -1) * trace_rec, dim=(0, 3))
        
        ## CHANGED: 출력 가중치의 그래디언트 계산에도 유사-미분이 곱해진 오차를 사용합니다.
        # trace_out의 차원은 [batch, n_hidden, time] -> brt
        # modulated_err의 차원은 [time, batch, n_out] -> tbo
        self.out.weight.grad += 0.05 * torch.einsum('tbo,brt->or', modulated_err, trace_out)

    def pseudo_derivative(self, v_membrane):
        return torch.max(torch.zeros_like(v_membrane), 1 - torch.abs(v_membrane - self.thr) / self.thr)

    def reset_parameters(self):
        self.init_net()
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5

class Basic_RSNN_eprop_forward(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = True,
        init_tau: float = 0.60,
        init_thresh = 0.6,  # Spike threshold
        init_tau_o = 0.6,
        gamma = 0.3,              # membrane decaying time constant    # spike trace decaying time constant
        width = 1
        
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.tau_o = init_tau_o
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = True
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        self.gamma = gamma
        self.width = width
        self.thr = init_thresh
        self.tau_o = init_tau_o
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        #self.fc1.weight.requires_grad = False

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        # init_thresh reaches the spiking nodes, not only self.thr (which is
        # used for the surrogate derivative). Without this the student always
        # fired at LIF_Node's default 0.5 regardless of init_thresh, so a
        # teacher built at any other threshold was a DIFFERENT network and its
        # weights were not a solution: planting them into the student gave
        # loss 0.848 instead of 0. With this wired in -- and the teacher built
        # from the same init_thresh/init_tau -- the teacher's own weights are
        # an exact optimum (loss 0.000000, verified for thresh 0.1..0.5).
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall(),
                             initial_thresh=init_thresh)
        self.out_node = LIF_Node(surrogate_function=HeavisideBoxcarCall(),
                                 initial_thresh=init_thresh)
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        torch.nn.init.kaiming_normal_(self.recurrent)

    def init_net(self):
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)


    def forward(self, x, label, training):
        self.device = x.device
        self.init_net()
        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)

        # Pre-h alpha-filtered traces
        trace_in_pre = torch.zeros(batch_size, self.n_in, device=self.device)
        trace_rec_pre = torch.zeros(batch_size, self.n_hidden, device=self.device)

        # Eligibility traces (kappa-filtered after multiplication with h_t)
        elig_in = torch.zeros(batch_size, self.n_hidden, self.n_in, device=self.device)
        elig_rec = torch.zeros(batch_size, self.n_hidden, self.n_hidden, device=self.device)

        # Output trace (kappa-filtered hidden spikes)
        trace_out_t = torch.zeros(batch_size, self.n_hidden, device=self.device)

        # z_{t-1} for trace_rec (matches minsik's [:, :, :n_t] slicing)
        prev_hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)

        for step in range(num_steps):
            input_spike = x[:, step, :]

            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.init_tau,
                    self.fc1(input_spike) + torch.mm(hidden_spike, self.recurrent)
                )
            else:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.init_tau, self.fc1(input_spike)
                )

            out_mem, out_spike = self.out_node(
                out_mem, out_spike, self.init_tau, self.out(hidden_spike)
            )

            err = out_spike - label[:, step, :]

            # Surrogate derivative: minsik uses init_tau * max(0, 1 - |v-thr|/thr)
            h_t = self.init_tau * torch.max(
                torch.zeros_like(hidden_mem),
                1 - torch.abs((hidden_mem - self.thr) / self.thr)
            )

            # Alpha-filtered pre-h traces
            # trace_in uses x[t]; trace_rec uses z[t-1] to match minsik
            trace_in_pre = self.init_tau * trace_in_pre + input_spike
            trace_rec_pre = self.init_tau * trace_rec_pre + prev_hidden_spike

            # Kappa-filtered eligibility traces
            elig_in = self.tau_o * elig_in + torch.einsum('br,bi->bri', h_t, trace_in_pre)
            elig_rec = self.tau_o * elig_rec + torch.einsum('br,bj->brj', h_t, trace_rec_pre)

            # Output trace (kappa-filtered z[t])
            trace_out_t = self.tau_o * trace_out_t + hidden_spike

            # Learning signal
            L = torch.einsum('bo,or->br', err, self.out.weight)

            # Gradient updates (factor 0.05 matches Basic_RSNN_eprop_minsik)
            self.fc1.weight.grad += 0.05 * torch.sum(L.unsqueeze(2) * elig_in, dim=0)
            self.recurrent.grad += 0.05 * torch.sum(L.unsqueeze(2) * elig_rec, dim=0)
            self.out.weight.grad += 0.05 * torch.einsum('bo,br->or', err, trace_out_t)

            prev_hidden_spike = hidden_spike.clone()

            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)

        return torch.stack(self.outputs, dim=1)

class Basic_RSNN_eprop_analog_forward(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = True,
        init_tau: float = 0.60,
        init_thresh = 0.6,  # Spike threshold
        init_tau_o = 0.6,
        gamma = 0.3,              # membrane decaying time constant    # spike trace decaying time constant
        width = 1,

        
    ):
        super().__init__()
        rpuconfig = SingleRPUConfig(device=ConstantStepDevice(dw_min=0.01))    
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.tau_o = init_tau_o
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = True
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        self.gamma = gamma
        self.width = width
        self.thr = init_thresh
        self.tau_o = init_tau_o

        self.fc1 = AnalogTile(self.n_hidden, self.n_in, bias=False, rpu_config=rpuconfig)
        weights_fc1, _ = self.fc1.get_weights()
        self.fc1.set_weights(weights_fc1/2)
        
        rec_inital_weights = (torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.recurrent = AnalogTile(self.n_hidden, self.n_hidden, bias=False, rpu_config=rpuconfig)
        self.recurrent.set_weights(rec_inital_weights)
        self.recurrent_analog_grad = torch.zeros_like(rec_inital_weights)


        self.out = AnalogTile(self.n_out, self.n_hidden, bias=False, rpu_config=rpuconfig)
        weights_out_analog, _ = self.out.get_weights()
        self.out.set_weights(weights_out_analog/2)
        self.out_analog_grad = torch.zeros_like(weights_out_analog)

        self.LIF0 = LIF_Node(surrogate_function=TriangleCall(gamma=self.gamma))
        self.out_node = LIF_Node(surrogate_function=TriangleCall(gamma=self.gamma))

    def init_net(self, device):
        self.fc1_grad = torch.zeros_like(self.fc1.get_weights()[0], device=device)
        self.recurrent_grad = torch.zeros_like(self.recurrent.get_weights()[0], device=device)
        self.out_grad = torch.zeros_like(self.out.get_weights()[0], device=device)          


    def forward(self, x, label, training):
        self.device = x.device
        self.init_net(self.device)                                       
        # x.shape = [batch_size, time, channel, width, height]
        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device = self.device)
        #effective_recurrent = self.recurrent * self.mask.to(self.device)
        #sparse_effective_recurrent=effective_recurrent*self.binary_tensor.to(self.device)
        
        trace_in_v = torch.zeros(batch_size, self.n_in, device=self.device)
        trace_rec_v = torch.zeros(batch_size, self.n_hidden, device=self.device)

        trace_out_t = torch.zeros(batch_size, self.n_hidden, device=self.device) 

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1.tile.forward(input_spike)+ self.recurrent.tile.forward(hidden_spike))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1.tile.forward(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out.tile.forward(hidden_spike))

            err = (out_spike - label[:,step,:])

            trace_in_v = self.init_tau * trace_in_v + input_spike
            trace_rec_v = self.init_tau * trace_rec_v + hidden_spike
            trace_out_t = self.tau_o * trace_out_t + hidden_spike
            h_t = self.gamma * torch.max(torch.zeros_like(hidden_mem), 1 - torch.abs((hidden_mem - self.thr) / self.thr))

            trace_in = torch.einsum('br,bi->bri', h_t, trace_in_v)
            trace_rec = torch.einsum('br,bi->bri', h_t, trace_rec_v)
            
            #L = torch.einsum('bo,or->br', err, self.out.weight)
            L = self.out.tile.backward(err) #L shape : b, r

            self.fc1_grad += 0.1 * torch.sum(L.unsqueeze(2) * trace_in, dim=(0))
            self.recurrent_grad += 0.1 * torch.sum(L.unsqueeze(2) * trace_rec, dim=(0))
            self.out_grad += 0.1 * torch.einsum('bo,br->or', err, trace_out_t)

            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)

            if step == (num_steps-1) and training:
                self.fc1.set_weights(self.fc1.get_weights()[0].to(self.device) - self.fc1_grad)
                self.recurrent.set_weights(self.recurrent.get_weights()[0].to(self.device) - self.recurrent_grad)
                self.out.set_weights(self.out.get_weights()[0].to(self.device) - self.out_grad)

        return torch.stack(self.outputs, dim=1)


class Basic_RSNN_eprop_aihwkit(nn.Module):
    def __init__(
        self,
        n_in = 100,
        n_hidden = 200,
        n_out = 20,
        subthresh = 0.5,
        recurrent = True,
        init_tau: float = 0.60,
        init_thresh = 0.6,  # Spike threshold
        init_tau_o = 0.6,
        gamma = 0.3,              # membrane decaying time constant    # spike trace decaying time constant
        width = 1
    ):
        super().__init__()
        
        rpuconfig = SingleRPUConfig(device=ConstantStepDevice(dw_min=0.01))    
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.tau_o = init_tau_o
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = True
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        self.gamma = gamma
        self.width = width
        self.thr = init_thresh

        self.fc1_analog = AnalogLinear(self.n_in, self.n_hidden, bias=False, rpu_config=rpuconfig)
        weights_fc1_analog, _ = self.fc1_analog.get_weights()
        self.fc1_analog.set_weights(weights_fc1_analog/2)
        self.fc1_analog_grad = torch.zeros_like(weights_fc1_analog)
        #init.kaiming_normal_(self.fc1.weight)
        #self.fc1.weight.data *= 0.5
        #self.fc1.weight.requires_grad = False
        
        
        
        
        rec_inital_weights = (torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.recurrent_analog = AnalogTile(self.n_hidden, self.n_hidden, bias=False, rpu_config=rpuconfig)
        self.recurrent_analog.set_weights(rec_inital_weights)
        self.recurrent_analog_grad = torch.zeros_like(rec_inital_weights)


        self.out_analog = AnalogLinear(self.n_hidden, self.n_out, bias=False, rpu_config=rpuconfig)
        weights_out_analog, _ = self.out_analog.get_weights()
        self.out_analog.set_weights(weights_out_analog/2)
        self.out_analog_grad = torch.zeros_like(weights_out_analog)
        #init.kaiming_normal_(self.out.weight)
        #self.out.weight.data *= 0.5
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        self.LIF0 = LIF_Node(surrogate_function=TriangleCall(gamma=self.gamma))
        self.out_node = LIF_Node(surrogate_function=TriangleCall(gamma=self.gamma))
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        #self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        #torch.nn.init.kaiming_normal_(self.recurrent)

        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        #self.fc1.weight.requires_grad = False

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden)/np.sqrt(self.n_hidden))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5
        #self.out.weight = nn.Parameter(torch.randn(self.n_hidden, self.n_out)/(np.sqrt(self.n_hidden)))
        self.LIF0 = LIF_Node(surrogate_function=TriangleCall(gamma=self.gamma))
        self.out_node = LIF_Node(surrogate_function=TriangleCall(gamma=self.gamma))
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        random_tensor = torch.rand(n_hidden, n_hidden)
        #self.binary_tensor = (random_tensor <= zero_ratio).float() * 0 + (random_tensor > zero_ratio).float() * 1
        torch.nn.init.kaiming_normal_(self.recurrent)


    def init_net(self):
        self.fc1_analog_grad = torch.zeros_like(self.fc1_analog.get_weights()[0]).to(self.device)
        self.recurrent_analog_grad = torch.zeros_like(self.recurrent_analog.get_weights()[0]).to(self.device)
        self.out_analog_grad = torch.zeros_like(self.out_analog.get_weights()[0]).to(self.device)

        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)

    def forward(self, x):
        self.device = x.device
        self.init_net()                                       
        # x.shape = [batch_size, time, channel, width, height]
        self.hidden_mem_analog_list = []
        self.hidden_spike_analog_list = []
        self.outputs_analog = []

        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        #x = x.view(x.size(0), x.size(1), -1)
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device = self.device)

        hidden_mem_analog = hidden_spike_analog = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem_analog = out_spike_analog = torch.zeros(batch_size, self.n_out, device = self.device)
        

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike, self.recurrent))
                hidden_mem_analog, hidden_spike_analog = self.LIF0(hidden_mem_analog, hidden_spike_analog, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike_analog, self.recurrent_analog.get_weights()[0].to(self.device)))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))
            out_mem_analog, out_spike_analog = self.out_node(out_mem_analog, out_spike_analog, self.init_tau, self.out_analog(hidden_spike_analog))
            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)

            self.outputs_analog.append(out_spike_analog)
            self.hidden_mem_analog_list.append(hidden_mem_analog)
            self.hidden_spike_analog_list.append(hidden_spike_analog)

        return torch.stack(self.outputs_analog, dim=1)


    def compute_grads(self, x, err):
        
        self.n_t = x.size(1)
        self.n_b = x.size(0)
        self.alpha = self.init_tau
        self.v = torch.stack(self.hidden_mem_list, dim=1).permute(1, 0, 2)
        self.z = torch.stack(self.hidden_spike_list, dim=1).permute(1, 0, 2)
        self.vo = torch.stack(self.outputs, dim=1).permute(1, 0, 2)

        # Surrogate derivatives
        h = self.gamma*torch.max(torch.zeros_like(self.v), 1-torch.abs((self.v-self.thr)/self.thr))
   
        alpha_conv  = torch.tensor([self.alpha ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_in    = F.conv1d(     x.permute(0,2,1), alpha_conv.expand(self.n_in ,-1,-1), padding=self.n_t, groups=self.n_in )[:,:,1:self.n_t+1].unsqueeze(1).expand(-1,self.n_hidden,-1,-1)  #n_b, n_rec, n_in , n_t 
        trace_in    = torch.einsum('tbr,brit->brit', h, trace_in )                                                                                                                          #n_b, n_rec, n_in , n_t 
        trace_rec   = F.conv1d(self.z.permute(1,2,0), alpha_conv.expand(self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_hidden)[:,:, :self.n_t  ].unsqueeze(1).expand(-1,self.n_hidden,-1,-1)  #n_b, n_rec, n_rec, n_t
        trace_rec   = torch.einsum('tbr,brit->brit', h, trace_rec)                                                                                                                          #n_b, n_rec, n_rec, n_t    
        trace_reg   = trace_rec

        # Output eligibility vector (vectorized computation, model-dependent)
        kappa_conv = torch.tensor([self.tau_o ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_out  = F.conv1d(self.z.permute(1,2,0), kappa_conv.expand(self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_hidden)[:,:,1:self.n_t+1]  #n_b, n_rec, n_t

        # Eligibility traces
        trace_in     = F.conv1d(   trace_in.reshape(self.n_b,self.n_in *self.n_hidden,self.n_t), kappa_conv.expand(self.n_in *self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_in *self.n_hidden)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_hidden,self.n_in ,self.n_t)   #n_b, n_rec, n_in , n_t  
        trace_rec    = F.conv1d(  trace_rec.reshape(self.n_b,self.n_hidden*self.n_hidden,self.n_t), kappa_conv.expand(self.n_hidden*self.n_hidden,-1,-1), padding=self.n_t, groups=self.n_hidden*self.n_hidden)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_hidden,self.n_hidden,self.n_t)   #n_b, n_rec, n_rec, n_t
        
        L = torch.einsum('tbo,or->brt', err, self.out.weight)
        #L = torch.einsum('tbo,or->brt', err, self.out.get_weights()[0].to(self.device))
        
        # Weight gradient updates
        self.fc1.state_dict()['analog_module.analog_tile_state']['analog_tile_weights'].grad+= 0.01*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * trace_in , dim=(0,3)) 
        self.recurrent_analog_grad += 0.01*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_hidden,-1) * trace_rec, dim=(0,3))
        self.out_analog.state_dict()['analog_module.analog_tile_state']['analog_tile_weights'].grad += 0.01*torch.einsum('tbo,brt->or', err, trace_out)

    def update_weights(self):
        self.fc1_analog.set_weights(self.fc1.get_weights()[0].to(self.device) - self.fc1_grad)
        self.recurrent_analog.set_weights(self.recurrent.get_weights()[0].to(self.device) - self.recurrent_grad)
        self.out_analog.set_weights(self.out.get_weights()[0].to(self.device) - self.out_grad)

    def pseudo_derivative(self, v_membrane):
        # Pseudo-derivative for spike generation (non-differentiability)
        # You can define it similar to the one in the paper (e.g., soft threshold function)
        return torch.max(torch.zeros_like(v_membrane), 1 - torch.abs(v_membrane - self.thr) / self.thr)


class Basic_RSNN_eprop_HW_forward(nn.Module):
    """
    E-prop SNN model with hardware-accelerated output layer gradient computation.

    This model uses a memristor crossbar array to compute the outer product
    for the output layer gradients. The hardware accumulates gradients over
    timesteps, and at epoch end, the accumulated gradient is read and applied
    to the software weights.

    Note: n_hidden and n_out must be 5 to match the 5x5 hardware array.
    """

    def __init__(
        self,
        n_in: int = 100,
        n_hidden: int = 5,  # Must be 5 for 5x5 hardware
        n_out: int = 5,     # Must be 5 for 5x5 hardware
        subthresh: float = 0.5,
        recurrent: bool = True,
        init_tau: float = 0.60,
        init_thresh: float = 0.6,
        init_tau_o: float = 0.6,
        gamma: float = 0.3,
        width: int = 1,
        # Hardware-specific parameters
        hw_enabled: bool = True,
        serial_port: str = 'COM7',
        baud_rate: int = 115200,
        bit_length: int = 10,
        use_mock_hw: bool = False,
        # When >0, the mock interface encodes each probability as a Bernoulli
        # bit stream of this length instead of using the exact product, so the
        # update is quantised to multiples of 1/bit_length. Isolates the
        # stochastic encoding from every other device effect.
        mock_quantize_bits: int = 0,
        mock_quantize_seed: int = 0,
        normalization_scale: float = 1.0,
        adc_to_grad_scale: float = 0.001,
        auto_calibrate_scale: bool = True,
        calibrate_ema: float = 0.5,
        # Pulse/read timing forwarded to MemristorInterface. Defaults are the
        # operating point validated by the 2026-08-05 uv_grid_sweep
        # (r ~ 0.92 at BL=10): width 15 us, read_time 20. The interface's own
        # defaults (width=1) were never validated and give ~1/10 the charge
        # per coincidence.
        pulse_width: int = 15,
        pulse_pre: int = 100,
        pulse_post: int = 100,
        pulse_zero: int = 10,
        read_time: int = 20,
        read_delay: int = 10,
        no_read_updates: bool = False,
        dno: bool = False,
    ):
        super().__init__()

        # Validate hardware constraints: the array is physically 5x5; a
        # smaller model maps onto its top-left block (rows = outputs,
        # columns = hidden). Used since the 2026-08-24 column-5 fault (4x4).
        if not (1 <= n_hidden <= 5) or not (1 <= n_out <= 5):
            raise ValueError(
                f"n_hidden and n_out must be in 1..5 for the 5x5 hardware. "
                f"Got n_hidden={n_hidden}, n_out={n_out}"
            )

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.tau_o = init_tau_o
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = True

        self.gamma = gamma
        self.width = width
        self.thr = init_thresh

        # Hardware settings
        self.hw_enabled = hw_enabled
        self.use_mock_hw = use_mock_hw
        self.normalization_scale = normalization_scale
        self.adc_to_grad_scale = adc_to_grad_scale

        # Auto-calibration of adc_to_grad_scale
        self.auto_calibrate_scale = auto_calibrate_scale
        self.calibrate_ema = calibrate_ema       # 0=keep old, 1=replace fully
        self._calibrated_once = False

        # Running max for normalization
        self.running_max_err = 1e-8
        self.running_max_trace = 1e-8

        # Initialize hardware interface
        if hw_enabled:
            from hardware import MemristorInterface, MockMemristorInterface
            if use_mock_hw:
                self.hw_interface = MockMemristorInterface(
                    port=serial_port,
                    baud_rate=baud_rate,
                    bit_length=bit_length,
                    quantize_bits=mock_quantize_bits,
                    quantize_seed=mock_quantize_seed,
                )
            else:
                self.hw_interface = MemristorInterface(
                    port=serial_port,
                    baud_rate=baud_rate,
                    bit_length=bit_length,
                    pulse_width=pulse_width,
                    pulse_pre=pulse_pre,
                    pulse_post=pulse_post,
                    pulse_zero=pulse_zero,
                    read_time=read_time,
                    read_delay=read_delay,
                    no_read_updates=no_read_updates,
                    dno=dno,
                )
        else:
            self.hw_interface = None

        # Network layers
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5

        self.recurrent = nn.Parameter(
            torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden)
        )
        torch.nn.init.kaiming_normal_(self.recurrent)

        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5

        # Same threshold-alignment fix as Basic_RSNN_eprop_forward: without
        # initial_thresh the node fires at LIF_Node's default 0.5 while
        # init_thresh only shapes the e-prop pseudo-derivative.
        self.LIF0 = LIF_Node(surrogate_function=HeavisideBoxcarCall(),
                             initial_thresh=init_thresh)
        self.out_node = LIF_Node(surrogate_function=HeavisideBoxcarCall(),
                                 initial_thresh=init_thresh)
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)

        # Optional learning window (t0, t1): when set, the error signal (and
        # therefore every weight update, hardware and software) is zeroed
        # outside these timesteps. For classification-style tasks whose
        # decision only reads a response window, spikes outside the window
        # are unconstrained -- forcing them to match the (all-zero) target
        # creates an unreachable objective and the training limit-cycles.
        # None (default) keeps the original behaviour.
        self.err_window = None

        # When True, per-timestep outer products are queued and flushed to
        # the hardware quadrant-major at apply_hw_gradient() time instead of
        # being sent immediately. This reduces P<->D command alternation on
        # shared lines from O(timesteps) to 3 per epoch, which the XOR runs
        # showed can otherwise cancel a mixed-sign column's accumulated
        # gradient via alternating half-select programming. Default False
        # keeps the original streaming behaviour.
        self.hw_batch_quadrants = False
        self._hw_queue = []

        # Optional fixed normalization constants (err_max, trace_max).
        # The default running-max normalization rescales each timestep by a
        # DIFFERENT factor as the max grows within the epoch, so the
        # accumulated outer product is a distorted version of the true
        # gradient sum -- measured on the perfect-raster teacher task this
        # distortion alone kept even the exact mock pipeline from converging
        # (best raster_err 3 vs 0). With fixed constants every timestep is
        # scaled identically and the accumulation is exact up to one global
        # factor, which auto-calibration absorbs. None keeps the original
        # running-max behaviour.
        self.fixed_norm = None

        # For gradient comparison (software vs hardware)
        self.desired_gradient_accumulated = torch.zeros(n_out, n_hidden)

    def connect_hardware(self) -> bool:
        """Connect to hardware. Call before training."""
        if self.hw_interface is not None:
            return self.hw_interface.connect()
        return False

    def disconnect_hardware(self):
        """Disconnect from hardware. Call after training."""
        if self.hw_interface is not None:
            self.hw_interface.disconnect()

    def reset_hardware(self, hard_reset: bool = True) -> bool:
        """Reset hardware state. Call at epoch start.

        Args:
            hard_reset: If True, send a physical Reset command to the device
                before measuring the new reference point. This pushes all
                cells back toward baseline conductance and prevents cumulative
                saturation across epochs.
        """
        if self.hw_interface is not None:
            success = self.hw_interface.reset(hard_reset=hard_reset)
            # Reset running normalization stats
            self.running_max_err = 1e-8
            self.running_max_trace = 1e-8
            self._hw_queue = []
            return success
        return False

    def init_net(self):
        """Initialize gradient buffers."""
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)

    def normalize_for_hardware(
        self,
        err: torch.Tensor,
        trace_out_t: torch.Tensor
    ) -> tuple:
        """
        Normalize err and trace_out_t to [0,1] probability range for hardware.

        Args:
            err: Error signal tensor of shape (batch, n_out)
            trace_out_t: Eligibility trace tensor of shape (batch, n_hidden)

        Returns:
            Tuple of (err_probs, trace_probs, err_signs, trace_signs)
            - err_probs: numpy array (n_out,) normalized to [0,1]
            - trace_probs: numpy array (n_hidden,) normalized to [0,1]
            - err_signs: tensor (n_out,) containing +1 or -1
            - trace_signs: tensor (n_hidden,) containing +1 or -1
        """
        # Average across batch
        err_avg = err.mean(dim=0).detach()  # (n_out,)
        trace_avg = trace_out_t.mean(dim=0).detach()  # (n_hidden,)

        # Extract signs for direction determination
        err_signs = torch.sign(err_avg)
        trace_signs = torch.sign(trace_avg)

        # Handle zero values (default to positive)
        err_signs[err_signs == 0] = 1
        trace_signs[trace_signs == 0] = 1

        # Take absolute values
        err_abs = torch.abs(err_avg)
        trace_abs = torch.abs(trace_avg)

        if self.fixed_norm is not None:
            err_max, trace_max = self.fixed_norm
        else:
            # Update running max
            self.running_max_err = max(self.running_max_err, err_abs.max().item())
            self.running_max_trace = max(self.running_max_trace, trace_abs.max().item())
            err_max = self.running_max_err
            trace_max = self.running_max_trace

        # Normalize to [0, 1]
        err_probs = (err_abs / err_max).clamp(0, 1)
        trace_probs = (trace_abs / trace_max).clamp(0, 1)

        # Apply scaling factor
        err_probs = err_probs * self.normalization_scale
        trace_probs = trace_probs * self.normalization_scale

        return (
            err_probs.cpu().numpy(),
            trace_probs.cpu().numpy(),
            err_signs,
            trace_signs
        )

    def send_to_hardware(
        self,
        err: torch.Tensor,
        trace_out_t: torch.Tensor
    ):
        """
        Send the per-timestep outer product err ⊗ trace_out_t to the memristor
        crossbar. The 4-quadrant sign decomposition is handled inside
        MemristorInterface.accumulate_outer_product().

        Args:
            err: Error signal tensor of shape (batch, n_out)
            trace_out_t: Eligibility trace tensor of shape (batch, n_hidden)
        """
        if self.hw_interface is None:
            return

        err_probs, trace_probs, err_signs, trace_signs = self.normalize_for_hardware(
            err, trace_out_t
        )

        u_p = err_probs.astype(np.float32)
        v_p = trace_probs.astype(np.float32)
        u_s = err_signs.detach().cpu().numpy().astype(np.float32)
        v_s = trace_signs.detach().cpu().numpy().astype(np.float32)
        # The array is physically 5x5; a smaller model (e.g. 4x4 after the
        # 2026-08-24 column-5 fault) uses the top-left block. Pad the unused
        # rows/columns with probability 0 so they never receive pulses
        # (single-line half-select is negligible, measured 2026-08-05).
        if u_p.size < 5:
            u_p = np.pad(u_p, (0, 5 - u_p.size))
            u_s = np.pad(u_s, (0, 5 - u_s.size), constant_values=1.0)
        if v_p.size < 5:
            v_p = np.pad(v_p, (0, 5 - v_p.size))
            v_s = np.pad(v_s, (0, 5 - v_s.size), constant_values=1.0)
        item = (u_p, v_p, u_s, v_s)
        if self.hw_batch_quadrants:
            self._hw_queue.append(item)
        else:
            self.hw_interface.accumulate_outer_product(*item)

    def forward(self, x, label, training):
        """
        Forward pass with optional hardware gradient accumulation.

        E-prop algorithm matches Basic_RSNN_eprop_forward exactly. The output
        layer gradient (err ⊗ trace_out_t outer product) is accumulated on the
        memristor crossbar each timestep when training and hardware is enabled.

        Args:
            x: Input tensor of shape (batch, time, n_in)
            label: Target tensor of shape (batch, time, n_out)
            training: Whether in training mode

        Returns:
            Output spike tensor of shape (batch, time, n_out)
        """
        self.device = x.device
        self.init_net()
        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)

        # Pre-h alpha-filtered traces
        trace_in_pre = torch.zeros(batch_size, self.n_in, device=self.device)
        trace_rec_pre = torch.zeros(batch_size, self.n_hidden, device=self.device)

        # Eligibility traces (kappa-filtered after multiplication with h_t)
        elig_in = torch.zeros(batch_size, self.n_hidden, self.n_in, device=self.device)
        elig_rec = torch.zeros(batch_size, self.n_hidden, self.n_hidden, device=self.device)

        # Output trace (kappa-filtered hidden spikes)
        trace_out_t = torch.zeros(batch_size, self.n_hidden, device=self.device)

        # z_{t-1} for trace_rec
        prev_hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)

        for step in range(num_steps):
            input_spike = x[:, step, :]

            if self.recurrent_connection:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.init_tau,
                    self.fc1(input_spike) + torch.mm(hidden_spike, self.recurrent)
                )
            else:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.init_tau, self.fc1(input_spike)
                )

            out_mem, out_spike = self.out_node(
                out_mem, out_spike, self.init_tau, self.out(hidden_spike)
            )

            err = out_spike - label[:, step, :]
            if self.err_window is not None and not \
                    (self.err_window[0] <= step < self.err_window[1]):
                err = torch.zeros_like(err)

            # Surrogate derivative: init_tau * max(0, 1 - |v-thr|/thr)
            h_t = self.init_tau * torch.max(
                torch.zeros_like(hidden_mem),
                1 - torch.abs((hidden_mem - self.thr) / self.thr)
            )

            # Alpha-filtered pre-h traces (trace_rec uses z[t-1])
            trace_in_pre = self.init_tau * trace_in_pre + input_spike
            trace_rec_pre = self.init_tau * trace_rec_pre + prev_hidden_spike

            # Kappa-filtered eligibility traces
            elig_in = self.tau_o * elig_in + torch.einsum('br,bi->bri', h_t, trace_in_pre)
            elig_rec = self.tau_o * elig_rec + torch.einsum('br,bj->brj', h_t, trace_rec_pre)

            # Output trace (kappa-filtered z[t])
            trace_out_t = self.tau_o * trace_out_t + hidden_spike

            # Learning signal
            L = torch.einsum('bo,or->br', err, self.out.weight)

            # Hidden-layer gradients (always software, factor 0.05 matches minsik)
            self.fc1.weight.grad += 0.05 * torch.sum(L.unsqueeze(2) * elig_in, dim=0)
            self.recurrent.grad += 0.05 * torch.sum(L.unsqueeze(2) * elig_rec, dim=0)

            # Output-layer gradient: outer product err ⊗ trace_out_t
            # Software-mirrored desired gradient (accumulated for HW comparison)
            desired_step_grad = 0.05 * torch.einsum('bo,br->or', err, trace_out_t)

            if training and self.hw_enabled and self.hw_interface is not None:
                # Send the per-timestep outer product to the memristor crossbar
                self.desired_gradient_accumulated += desired_step_grad
                self.send_to_hardware(err, trace_out_t)
            else:
                # Software fallback: accumulate directly into out.weight.grad
                self.out.weight.grad += desired_step_grad

            prev_hidden_spike = hidden_spike.clone()

            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)

        return torch.stack(self.outputs, dim=1)

    def apply_hw_gradient(self, learning_rate: float = 0.01):
        """
        Apply accumulated hardware gradient to output weights.

        Call at the end of each epoch to:
        1. Read accumulated gradient from hardware
        2. Convert ADC values to gradient scale
        3. Compare with desired (software) gradient
        4. Apply to software weights

        Args:
            learning_rate: Learning rate for weight update
        """
        if not self.hw_enabled or self.hw_interface is None:
            return

        # Ablation switch: freeze W_out at its init values. In HW mode the
        # forward pass never touches out.weight.grad (gradients go to the
        # accumulator), so skipping this method leaves the output layer
        # completely untrained while fc1/recurrent still learn through the
        # optimizer. Used to isolate how much of the loss drop the
        # (analog-trained) output layer actually contributes.
        if getattr(self, 'freeze_wout', False):
            return

        # Flush queued outer products quadrant-major (no-op if not batching)
        if self.hw_batch_quadrants and self._hw_queue:
            self.hw_interface.accumulate_outer_products_grouped(self._hw_queue)
            self._hw_queue = []

        # Read accumulated gradient from hardware; the physical array is
        # (5, 5) -- a smaller model reads its top-left (n_out, n_hidden) block
        hw_gradient_adc = self.hw_interface.read_accumulated_gradient()
        hw_gradient_adc = hw_gradient_adc[: self.n_out, : self.n_hidden]

        # Convert ADC to gradient scale
        hw_gradient = torch.tensor(
            hw_gradient_adc * self.adc_to_grad_scale,
            dtype=torch.float32,
            device=self.device
        )

        # Compare with desired gradient
        import logging
        logger = logging.getLogger(__name__)

        logger.info("\n" + "="*70)
        logger.info("GRADIENT COMPARISON: Desired (Software) vs Hardware")
        logger.info("="*70)

        logger.info("\n[Desired Gradient (Software)]:")
        logger.info(f"{self.desired_gradient_accumulated.detach().cpu().numpy()}")

        logger.info("\n[Hardware Gradient (Raw ADC)]:")
        logger.info(f"{hw_gradient_adc}")

        logger.info("\n[Hardware Gradient (Scaled)]:")
        logger.info(f"{hw_gradient.detach().cpu().numpy()}")

        # Calculate difference
        diff = self.desired_gradient_accumulated - hw_gradient
        logger.info("\n[Difference (Desired - Hardware)]:")
        logger.info(f"{diff.detach().cpu().numpy()}")

        # Calculate statistics
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        corr = torch.corrcoef(torch.stack([
            self.desired_gradient_accumulated.flatten(),
            hw_gradient.flatten()
        ]))[0, 1].item() if torch.sum(hw_gradient**2) > 0 else 0.0

        logger.info(f"\n[Statistics]:")
        logger.info(f"  MSE (Mean Squared Error): {mse:.6f}")
        logger.info(f"  MAE (Mean Absolute Error): {mae:.6f}")
        logger.info(f"  Correlation Coefficient: {corr:.4f}")
        logger.info("="*70 + "\n")

        # Auto-calibrate adc_to_grad_scale by matching mean magnitudes.
        # We want:  scale * mean|hw_adc|  ≈  mean|sw_grad|
        # so:       scale_target = mean|sw_grad| / mean|hw_adc|
        # First pass replaces fully; later passes use EMA for stability.
        if self.auto_calibrate_scale:
            sw_mag = float(torch.abs(self.desired_gradient_accumulated).mean().item())
            hw_mag = float(np.abs(hw_gradient_adc).mean())
            if hw_mag > 1e-9 and sw_mag > 1e-9:
                scale_target = sw_mag / hw_mag
                if not self._calibrated_once:
                    new_scale = scale_target
                    self._calibrated_once = True
                else:
                    a = self.calibrate_ema
                    new_scale = (1 - a) * self.adc_to_grad_scale + a * scale_target
                logger.info(
                    f"[CALIBRATE] sw_mag={sw_mag:.4f}, hw_adc_mag={hw_mag:.4f}, "
                    f"target_scale={scale_target:.6f}, "
                    f"adc_to_grad_scale {self.adc_to_grad_scale:.6f} -> {new_scale:.6f}"
                )
                self.adc_to_grad_scale = new_scale
                # Recompute hw_gradient with the updated scale before the
                # weight update so this epoch already benefits from calibration
                hw_gradient = torch.tensor(
                    hw_gradient_adc * self.adc_to_grad_scale,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                logger.info(
                    f"[CALIBRATE] skipped (sw_mag={sw_mag:.2e}, hw_adc_mag={hw_mag:.2e})"
                )

        # Optional per-column gain calibration on top of the global scale.
        # Measured on the XOR runs (2026-08-10): stable per-column gain
        # spread of ~3.7x (weakest column 0.6, strongest 2.2 relative to
        # desired), i.e. the effective learning rate differs per hidden
        # neuron. A single global scale cannot equalize this; five EMA
        # scalars can. Enable with model.calibrate_per_column = True.
        if getattr(self, 'calibrate_per_column', False):
            sw_col = self.desired_gradient_accumulated.abs().mean(dim=0)
            sw_col = sw_col.detach().cpu().numpy()          # (n_hidden,)
            hw_col = np.abs(hw_gradient_adc).mean(axis=0)   # (n_hidden,)
            if not hasattr(self, '_col_gain') or self._col_gain is None:
                self._col_gain = np.ones(self.n_hidden, dtype=np.float64)
            for j in range(self.n_hidden):
                if hw_col[j] > 1e-9 and sw_col[j] > 1e-9:
                    # relative gain vs the global scale already applied
                    target = (sw_col[j] / hw_col[j]) / self.adc_to_grad_scale
                    a = self.calibrate_ema
                    self._col_gain[j] = ((1 - a) * self._col_gain[j]
                                         + a * target)
            # clamp: never boost a column more than 5x or cut below 0.2x,
            # so a noise-floor column cannot blow up the update
            col_gain = np.clip(self._col_gain, 0.2, 5.0)
            logger.info(f"[CALIBRATE-COL] gains: {np.round(col_gain, 3)}")
            hw_gradient = torch.tensor(
                hw_gradient_adc * self.adc_to_grad_scale * col_gain[None, :],
                dtype=torch.float32, device=self.device,
            )

        # Per-cell record of what the algorithm asked for versus what the
        # crossbar returned, appended once per epoch. The text log above
        # prints the same matrices but cannot be analysed afterwards; this
        # keeps every one of the 25 weights so correlation, per-cell gain and
        # drift over epochs can be reconstructed.
        if getattr(self, "grad_log_path", None):
            import csv as _csv
            import os as _os
            desired_np = self.desired_gradient_accumulated.detach().cpu().numpy()
            hw_np = hw_gradient.detach().cpu().numpy()
            new_file = not _os.path.exists(self.grad_log_path)
            with open(self.grad_log_path, "a", newline="",
                      encoding="utf-8") as _f:
                w = _csv.writer(_f)
                if new_file:
                    w.writerow(["epoch", "row", "col", "desired", "hw_adc",
                                "hw_scaled", "adc_to_grad_scale"])
                ep = getattr(self, "grad_log_epoch", 0)
                for i in range(desired_np.shape[0]):
                    for j in range(desired_np.shape[1]):
                        w.writerow([ep, i + 1, j + 1,
                                    float(desired_np[i, j]),
                                    float(hw_gradient_adc[i, j]),
                                    float(hw_np[i, j]),
                                    float(self.adc_to_grad_scale)])
            self.grad_log_epoch = ep + 1

        # Reset desired gradient accumulator for next epoch
        self.desired_gradient_accumulated.zero_()

        # Apply to software weights (gradient descent)
        weight_before = self.out.weight.data.clone()
        with torch.no_grad():
            self.out.weight.data -= learning_rate * hw_gradient
        weight_change = (self.out.weight.data - weight_before).abs().max().item()

        logger.info(f"\n[WEIGHT UPDATE] Learning rate={learning_rate}, Max weight change={weight_change:.6f}")
        logger.info(f"[WEIGHT UPDATE] Weight range: [{self.out.weight.data.min():.4f}, {self.out.weight.data.max():.4f}]")

        # Returned so callers can feed the hardware gradient into their own
        # optimizer instead (pass learning_rate=0 to skip the direct write).
        return hw_gradient

    def get_hw_gradient(self) -> np.ndarray:
        """Get the current accumulated gradient from hardware (for debugging)."""
        if self.hw_interface is not None:
            return self.hw_interface.read_accumulated_gradient()
        return np.zeros((5, 5))