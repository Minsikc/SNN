"""Same transpose test against the LOCAL Basic_RSNN_eprop_forward."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from models.models import Basic_RSNN_eprop_forward

torch.manual_seed(0)
n_in, H, n_out, T, B = 10, 8, 4, 30, 3

m = Basic_RSNN_eprop_forward(n_in=n_in, n_hidden=H, n_out=n_out,
                             recurrent=True, init_thresh=0.5)
# widen the Boxcar surrogate so the BPTT reference gradient is informative
# (repo default +-0.1 passes almost no gradient -> meaningless correlations)
m.LIF0.surrogate_function.subthresh = torch.tensor(0.5)
m.out_node.surrogate_function.subthresh = torch.tensor(0.5)
x = (torch.rand(B, T, n_in) < 0.3).float()
y = (torch.rand(B, T, n_out) < 0.2).float()

out = m(x, y, training=True)
G_ep_rec = m.recurrent.grad.clone()
G_ep_out = m.out.weight.grad.clone()
G_ep_fc1 = m.fc1.weight.grad.clone()

out = m(x, y, training=False)
m.init_net()
loss = 0.5 * ((out - y) ** 2).sum()
loss.backward()
G_bp_rec = m.recurrent.grad.clone()
G_bp_out = m.out.weight.grad.clone()
G_bp_fc1 = m.fc1.weight.grad.clone()


def corr(a, b):
    a, b = a.flatten(), b.flatten()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


print(f"fc1 : corr(eprop, bptt)   = {corr(G_ep_fc1, G_bp_fc1):+.3f}")
print(f"out : corr(eprop, bptt)   = {corr(G_ep_out, G_bp_out):+.3f}")
print(f"rec : corr(eprop, bptt)   = {corr(G_ep_rec, G_bp_rec):+.3f}")
print(f"rec : corr(eprop, bptt.T) = {corr(G_ep_rec, G_bp_rec.t()):+.3f}")
