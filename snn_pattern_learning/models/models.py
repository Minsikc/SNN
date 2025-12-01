import torch
import torch.nn as nn
from neurons import LIF_Node, PLIF_Node, HeavisideBoxcarCall, Accumulate_node, TriangleCall
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init

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
        self.fc1.weight.data *= 0.5

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
            self.grads_batch(x, yo, yt)
            
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
        self.LIF0 = LIF_Node(surrogate_function=TriangleCall())
        self.out_node = LIF_Node(surrogate_function=TriangleCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        torch.nn.init.kaiming_normal_(self.recurrent)

    def init_net(self):
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)


    def forward(self, x, label, training):
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
        
        trace_in_v = torch.zeros(batch_size, self.n_in, device=self.device)
        trace_rec_v = torch.zeros(batch_size, self.n_hidden, device=self.device)

        trace_out_t = torch.zeros(batch_size, self.n_hidden, device=self.device) 

        for step in range(num_steps):
            input_spike = x[:, step,:]
            if self.recurrent_connection is True:
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike)+torch.mm(hidden_spike, self.recurrent))
            else : 
                hidden_mem, hidden_spike = self.LIF0(hidden_mem, hidden_spike, self.init_tau,self.fc1(input_spike))

            out_mem, out_spike = self.out_node(out_mem, out_spike, self.init_tau, self.out(hidden_spike))

            err = (out_spike - label[:,step,:])

            trace_in_v = self.init_tau * trace_in_v + input_spike
            trace_rec_v = self.init_tau * trace_rec_v + hidden_spike
            trace_out_t = self.tau_o * trace_out_t + hidden_spike
            h_t = self.gamma * torch.max(torch.zeros_like(hidden_mem), 1 - torch.abs((hidden_mem - self.thr) / self.thr))

            trace_in = torch.einsum('br,bi->bri', h_t, trace_in_v)
            trace_rec = torch.einsum('br,bi->bri', h_t, trace_rec_v)
            
            L = torch.einsum('bo,or->br', err, self.out.weight)

            self.fc1.weight.grad += 0.01 * torch.sum(L.unsqueeze(2) * trace_in, dim=(0))
            self.recurrent.grad += 0.01 * torch.sum(L.unsqueeze(2) * trace_rec, dim=(0))
            self.out.weight.grad += 0.01 * torch.einsum('bo,br->or', err, trace_out_t)

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