"""Diagnose spiking activity of the init-state model on TemporalXORDataset."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets.customdatasets import TemporalXORDataset
from models.models import Basic_RSNN_eprop_forward

ds = TemporalXORDataset()
x, tgt = ds.data, ds.targets

for seed in range(4):
    torch.manual_seed(seed)
    m = Basic_RSNN_eprop_forward(n_in=10, n_hidden=5, n_out=5, recurrent=True,
                                 init_tau=0.6, init_thresh=0.2, init_tau_o=0.6,
                                 gamma=0.3)
    with torch.no_grad():
        out = m(x, tgt, training=False)
    hid = torch.stack(m.hidden_spike_list, dim=1)  # (4, T, 5)
    print(f"seed {seed}: hidden spikes/sample {hid.sum(dim=(1,2)).tolist()}  "
          f"out spikes/sample {out.sum(dim=(1,2)).tolist()}")
    # membrane stats of output neurons in response window
    print(f"   fc1 |w| mean {m.fc1.weight.abs().mean():.3f}  "
          f"out |w| mean {m.out.weight.abs().mean():.3f}")
