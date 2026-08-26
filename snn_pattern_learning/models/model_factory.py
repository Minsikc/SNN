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
            recurrent=model_config.get('recurrent', False),
            init_tau=model_config.get('init_tau', 0.6),
            weight_scale=model_config.get('weight_scale', 0.5)
        )
    elif model_type == "RSNN_eprop":
        model = Basic_RSNN_eprop_minsik(
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            subthresh=model_config.get('subthresh', 0.5),
            recurrent=model_config.get('recurrent', True),
            init_tau=model_config.get('init_tau', 0.60),
            init_thresh=model_config.get('init_thresh', 0.6),
            init_tau_o=model_config.get('init_tau_o', 0.6),
            gamma=model_config.get('gamma', 0.3),
            width=model_config.get('width', 1)
        )
    elif model_type == "RSNN_eprop_forward":
        model = Basic_RSNN_eprop_forward(
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            recurrent=model_config.get('recurrent', True),
            init_tau=model_config.get('init_tau', 0.60),
            init_thresh=model_config.get('init_thresh', 0.5),
            init_tau_o=model_config.get('init_tau_o', 0.6),
            gamma=model_config.get('gamma', 0.3),
            width=model_config.get('width', 1)
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
            # Neuron parameters were previously NOT forwarded, so the model
            # silently ran at constructor defaults (init_thresh=0.6 after the
            # threshold-alignment fix) no matter what the yaml said, while
            # the teacher dataset DID receive model.init_thresh -- breaking
            # teacher/student threshold alignment. Now wired through.
            init_tau=model_config.get('init_tau', 0.60),
            init_thresh=model_config.get('init_thresh', 0.5),
            init_tau_o=model_config.get('init_tau_o', 0.6),
            gamma=model_config.get('gamma', 0.3),
            hw_enabled=hw_config.get('enabled', True),
            serial_port=hw_config.get('serial_port', 'COM7'),
            baud_rate=hw_config.get('baud_rate', 115200),
            bit_length=hw_config.get('bit_length', 10),
            use_mock_hw=hw_config.get('use_mock_hw', False),
            mock_quantize_bits=hw_config.get('mock_quantize_bits', 0),
            mock_quantize_seed=hw_config.get('mock_quantize_seed', 0),
            adc_to_grad_scale=hw_config.get('adc_to_grad_scale', 0.001),
            auto_calibrate_scale=hw_config.get('auto_calibrate_scale', True),
            calibrate_ema=hw_config.get('calibrate_ema', 0.5),
            # these were previously silently dropped, so the interface always
            # ran at its own defaults (pulse_width=1) regardless of the yaml
            normalization_scale=hw_config.get('normalization_scale', 1.0),
            pulse_width=hw_config.get('pulse_width', 15),
            pulse_pre=hw_config.get('pulse_pre', 100),
            pulse_post=hw_config.get('pulse_post', 100),
            pulse_zero=hw_config.get('pulse_zero', 10),
            read_time=hw_config.get('read_time', 20),
            read_delay=hw_config.get('read_delay', 10),
            no_read_updates=hw_config.get('no_read_updates', False),
            dno=hw_config.get('dno', False),
        )
        # ablation: freeze the output layer (skip apply_hw_gradient)
        model.freeze_wout = hw_config.get('freeze_wout', False)
        # per-epoch CSV of desired vs hardware-read gradient, all 25 cells
        model.grad_log_path = hw_config.get('grad_log_path', None)
        model.grad_log_epoch = 0
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Patch neuron functions ONLY when the yaml explicitly configures the
    # neuron (non-empty config). Previously the default neuron_type
    # 'triangular' with an empty config silently replaced every model's own
    # surrogate (e.g. the HW model's Boxcar) with a default TriangleCall.
    if neuron_type is not None and neuron_config:
        neuron_function = create_neuron_function(neuron_type, neuron_config)
        model = patch_model_neurons(model, neuron_function)

    return model


class ConfigurableBasicRSNN(nn.Module):
    """
    Configurable version of Basic_RSNN_spike that allows different neuron types
    """
    def __init__(self, n_in=100, n_hidden=200, n_out=20, subthresh=0.5,
                 recurrent=False, init_tau=0.60, weight_scale=0.5,
                 neuron_type='triangular', neuron_config=None):
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

        # Create neuron function based on type
        if neuron_config is None:
            neuron_config = {}

        neuron_function = create_neuron_function(neuron_type, neuron_config)

        # Initialize layers
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= self.weight_scale

        self.recurrent = nn.Parameter(torch.rand(self.n_hidden, self.n_hidden) / torch.sqrt(torch.tensor(self.n_hidden)))
        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        nn.init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= self.weight_scale
        
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
            init_tau=model_config.get('init_tau', 0.6),
            weight_scale=model_config.get('weight_scale', 0.5),
            neuron_type=neuron_type,
            neuron_config=neuron_config
        )
    else:
        # For other model types, use the patching method
        return create_model(model_type, model_config, neuron_type, neuron_config)