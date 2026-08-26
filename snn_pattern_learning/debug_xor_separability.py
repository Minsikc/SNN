"""Is XOR linearly readable from the frozen random reservoir's traces?

For each seed: roll out the 5-neuron reservoir (fixed random fc1/recurrent)
on the 4 XOR patterns, take the kappa-filtered hidden trace at each response
step, and fit a least-squares linear readout mapping trace -> target spikes.
If the fit classifies all 4 patterns correctly, an ideal W_out exists and the
crossbar-trained readout has something to find.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets.customdatasets import TemporalXORDataset
from models.models import Basic_RSNN_eprop_forward
from run_xor import MODEL_KW

def reservoir_traces(model, x, tau_o):
    """Hidden spikes and kappa traces for every step. x: (B,T,10)"""
    B, T, _ = x.shape
    mem = spk = torch.zeros(B, 5)
    trace = torch.zeros(B, 5)
    traces = []
    with torch.no_grad():
        for t in range(T):
            mem, spk = model.LIF0(mem, spk, model.init_tau,
                                  model.fc1(x[:, t]) + spk @ model.recurrent)
            trace = tau_o * trace + spk
            traces.append(trace.clone())
    return torch.stack(traces, 1)   # (B, T, 5)

def test(seed, tau, tau_o, gap):
    if gap == "long":
        ds = TemporalXORDataset(a_window=(1, 4), b_window=(9, 12),
                                response_window=(14, 20))
    else:
        ds = TemporalXORDataset(a_window=(1, 4), b_window=(5, 8),
                                response_window=(10, 16))
    torch.manual_seed(seed)
    m = Basic_RSNN_eprop_forward(**{**MODEL_KW, "init_tau": tau,
                                    "init_tau_o": tau_o})
    tr = reservoir_traces(m, ds.data, tau_o)
    r0, r1 = ds.response_window
    X = tr[:, r0:r1, :].reshape(-1, 5)            # (4*W, 5)
    Y = ds.targets[:, r0:r1, :].reshape(-1, 5)    # (4*W, 5)
    # least-squares readout (with small ridge)
    A = X.T @ X + 1e-3 * torch.eye(5)
    W = torch.linalg.solve(A, X.T @ Y)            # (5, 5)
    pred = (X @ W).reshape(4, r1 - r0, 5)
    # classify: group {0,1} vs {3,4} mean prediction over window
    g0 = pred[:, :, 0:2].sum(dim=(1, 2))
    g1 = pred[:, :, 3:5].sum(dim=(1, 2))
    cls = (g1 > g0).long()
    labels = torch.tensor([0, 1, 1, 0])
    acc = (cls == labels).float().mean().item()
    # how distinct are the 4 patterns' mean traces?
    mean_tr = tr[:, r0:r1, :].mean(dim=1)         # (4, 5)
    d01 = (mean_tr[0] - mean_tr[1]).norm().item()
    return acc, mean_tr, d01

print(f"{'seed':>4} {'tau':>4} {'tau_o':>5} {'gap':>5} {'linsep acc':>10}")
for gap in ["long", "short"]:
    for tau in [0.6, 0.8, 0.9]:
        for tau_o in [0.6, 0.9]:
            accs = []
            for seed in range(8):
                acc, mt, _ = test(seed, tau, tau_o, gap)
                accs.append(acc)
            ok = sum(a == 1.0 for a in accs)
            print(f"tau={tau} tau_o={tau_o} gap={gap:5s}  "
                  f"perfect {ok}/8  accs={[f'{a:.2f}' for a in accs]}")
