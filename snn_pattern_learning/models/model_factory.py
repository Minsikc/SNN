import torch
import torch.nn as nn
import copy
from .models import (
    Basic_RSNN_spike, Basic_RSNN_eprop_minsik, Basic_RSNN_eprop_forward,
    Basic_RSNN_eprop_analog_forward, RSNN_merged_hidden_output, RSNN_fixed_w_in,
    Basic_RSNN_eprop_HW_forward
)
from neurons import HeavisideBoxcarCall, TriangleCall, LIF_Node


class NeuronWrapper(nn.Module):
    """Wrapper to make neuron functions compatible with LIF_Node"""
    def __init__(self, neuron_function):
        super().__init__()
        self.neuron_function = neuron_function
        
    def forward(self, mem, thresh=None):
        # Handle different neuron function signatures
        if hasattr(self.neuron_function, 'forward'):
            import inspect
            sig = inspect.signature(self.neuron_function.forward)
            param_names = list(sig.parameters.keys())
            
            # Check if thresh is a parameter (excluding 'self')
            if 'thresh' in param_names:
                return self.neuron_function(mem, thresh)
            else:
                return self.neuron_function(mem)
        else:
            return self.neuron_function(mem)

def create_neuron_function(neuron_type, neuron_config):
    """Create neuron function based on type and config"""
    if neuron_type == "triangular":
        base_function = TriangleCall(
            thresh=neuron_config.get('thresh', 0.6),
            subthresh=neuron_config.get('subthresh', 0.25),
            gamma=neuron_config.get('gamma', 0.3),
            width=neuron_config.get('width', 1)
        )
    elif neuron_type == "boxcar":
        base_function = HeavisideBoxcarCall(
            thresh=neuron_config.get('thresh', 0.4),
            subthresh=neuron_config.get('subthresh', 0.1),
            alpha=neuron_config.get('alpha', 1.0)
        )
    else:
        raise ValueError(f"Unsupported neuron type: {neuron_type}")
    
    return NeuronWrapper(base_function)


def patch_model_neurons(model, neuron_function):
    """Patch model's neuron functions with the specified type"""
    for name, module in model.named_modules():
        if isinstance(module, LIF_Node):
            module.surrogate_function = neuron_function
    return model


def create_model(model_type, model_config, neuron_type=None, neuron_config=None):
    """
    Create model with specified neuron type
    
    Args:
        model_type: Type of model (e.g., 'Basic_RSNN_spike')
        model_config: Model configuration (n_in, n_hidden, n_out, etc.)
        neuron_type: Type of neuron ('triangular' or 'boxcar')
        neuron_config: Neuron-specific configuration
    
    Returns:
        Model instance with patched neuron functions
    """
    n_in = model_config.get('n_in', 50)
    n_hidden = model_config.get('n_hidden', 40)
    n_out = model_config.get('n_out', 10)
    
    # Create base model
    if model_type == "Basic_RSNN_spike":
        model = Basic_RSNN_spike(
            n_in=n_in, 
            n_hidden=n_hidden, 
            n_out=n_out,
            recurrent=model_config.get('recurrent', False)
        )
    elif model_type == "RSNN_eprop":
        model = Basic_RSNN_eprop_minsik(
            n_in=n_in, 
            n_hidden=n_hidden, 
            n_out=n_out
        )
    elif model_type == "RSNN_eprop_forward":
        model = Basic_RSNN_eprop_forward(
            n_in=n_in, 
            n_hidden=n_hidden, 
            n_out=n_out
        )
    elif model_type == "RSNN_eprop_analog_forward":
        model = Basic_RSNN_eprop_analog_forward(
            n_in=n_in, 
            n_hidden=n_hidden, 
            n_out=n_out, 
            recurrent=model_config.get('recurrent', True)
        )
    elif model_type == "RSNN_merged_hidden_output":
        model = RSNN_merged_hidden_output(
            n_in=n_in, 
            n_hidden=n_hidden, 
            n_out=n_out
        )
    elif model_type == "RSNN_fixed_w_in":
        model = RSNN_fixed_w_in(
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out
        )
    elif model_type == "RSNN_eprop_HW_forward":
        # Hardware-integrated model - requires special handling
        hw_config = model_config.get('hardware', {})
        model = Basic_RSNN_eprop_HW_forward(
            n_in=n_in,
            n_hidden=n_hidden,  # Must be 5 for hardware
            n_out=n_out,        # Must be 5 for hardware
            recurrent=model_config.get('recurrent', True),
            hw_enabled=hw_config.get('enabled', True),
            serial_port=hw_config.get('serial_port', 'COM7'),
            baud_rate=hw_config.get('baud_rate', 115200),
            bit_length=hw_config.get('bit_length', 10),
            use_mock_hw=hw_config.get('use_mock', False),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Patch neuron functions if specified
    if neuron_type is not None and neuron_config is not None:
        neuron_function = create_neuron_function(neuron_type, neuron_config)
        model = patch_model_neurons(model, neuron_function)
    
    return model


class ConfigurableBasicRSNN(nn.Module):
    """
    Configurable version of Basic_RSNN_spike that allows different neuron types
    """
    def __init__(self, n_in=100, n_hidden=200, n_out=20, subthresh=0.5, 
                 recurrent=False, init_tau=0.60, neuron_type='triangular', neuron_config=None):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.recurrent_connection = recurrent
        self.custom_grad = False
        self.custom_grad_forward = False
        
        # Create neuron function based on type
        if neuron_config is None:
            neuron_config = {}
        
        neuron_function = create_neuron_function(neuron_type, neuron_config)
        
        # Initialize layers
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5
        
        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / torch.sqrt(torch.tensor(self.n_hidden)))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        nn.init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5
        
        # Initialize neuron nodes with specified function
        # Use a copy of the neuron function for each node
        self.LIF0 = LIF_Node(surrogate_function=copy.deepcopy(neuron_function))
        self.out_node = LIF_Node(surrogate_function=copy.deepcopy(neuron_function))
        
        # Create mask for recurrent connections
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)
        nn.init.kaiming_normal_(self.recurrent)
    
    def forward(self, x):
        self.device = x.device
        outputs = []
        
        num_steps = x.size(1)
        batch_size = x.size(0)
        
        # Initialize states
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)
        effective_recurrent = self.recurrent * self.mask.to(self.device)
        
        for step in range(num_steps):
            input_spike = x[:, step, :]
            
            if self.recurrent_connection:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.init_tau,
                    self.fc1(input_spike) + torch.mm(hidden_spike, effective_recurrent)
                )
            else:
                hidden_mem, hidden_spike = self.LIF0(
                    hidden_mem, hidden_spike, self.init_tau,
                    self.fc1(input_spike)
                )
            
            out_mem, out_spike = self.out_node(
                out_mem, out_spike, self.init_tau, 
                self.out(hidden_spike)
            )
            outputs.append(out_spike)
        
        return torch.stack(outputs, dim=1)


def create_configurable_model(model_type, model_config, neuron_type='triangular', neuron_config=None):
    """
    Create a configurable model with specified neuron type
    
    Args:
        model_type: Type of model
        model_config: Model configuration
        neuron_type: Type of neuron ('triangular' or 'boxcar')
        neuron_config: Neuron-specific configuration
    
    Returns:
        Configurable model instance
    """
    if neuron_config is None:
        neuron_config = {}
    
    if model_type == "Basic_RSNN_spike":
        return ConfigurableBasicRSNN(
            n_in=model_config.get('n_in', 50),
            n_hidden=model_config.get('n_hidden', 40),
            n_out=model_config.get('n_out', 10),
            recurrent=model_config.get('recurrent', False),
            neuron_type=neuron_type,
            neuron_config=neuron_config
        )
    else:
        # For other model types, use the patching method
        return create_model(model_type, model_config, neuron_type, neuron_config)