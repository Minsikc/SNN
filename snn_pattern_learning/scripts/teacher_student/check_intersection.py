"""Find a model seed where BOTH rules reach a perfect raster on T=12/seed11.

(a) BPTT mseed=0 long runs (best=1 at 1000 ep -- does it close?)
(b) e-prop at more model seeds / finer lr
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
import numpy as np
import torch
from sweep_teacher_perfect import build_task, raster_err
from models.models import Basic_RSNN_eprop_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time

T, DS_SEED = 12, 11


def run_rule(rule, lr, mseed, epochs, subthresh=0.5):
    x, tgt = build_task(T, DS_SEED)
    torch.manual_seed(mseed)
    m = Basic_RSNN_eprop_forward(n_in=10, n_hidden=5, n_out=5,
                                 recurrent=True, init_thresh=0.2)
    if rule == "bptt":
        m.custom_grad = False
        m.custom_grad_forward = False
        m.LIF0.surrogate_function.subthresh = torch.tensor(subthresh)
        m.out_node.surrogate_function.subthresh = torch.tensor(subthresh)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    kernel = create_exponential_kernel(3, 2.0)
    first, flags, best = -1, [], 99
    for ep in range(epochs):
        opt.zero_grad()
        out = m(x, tgt, training=True)
        co = apply_convolution(out, kernel, 3)
        ct = apply_convolution(tgt, kernel, 3)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1])
        if rule == "bptt":
            m.init_net()
            loss.backward()
        opt.step()
        with torch.no_grad():
            o = m(x, tgt, training=False)
        err = sum(raster_err(o, tgt))
        best = min(best, err)
        flags.append(err == 0)
        if flags[-1] and first < 0:
            first = ep + 1
    return first, float(np.mean(flags[-50:])), best


print("--- (a) BPTT mseed=0, long ---")
for lr in [0.005, 0.01]:
    f, h, b = run_rule("bptt", lr, 0, 3000)
    print(f"bptt mseed=0 lr={lr} 3000ep: first@{f} hold {h:.2f} best {b}",
          flush=True)

print("--- (b) e-prop, more seeds ---")
for ms in [1, 2, 3, 4, 5, 6, 7]:
    for lr in [0.08, 0.1, 0.15]:
        f, h, b = run_rule("digital", lr, ms, 300)
        if f > 0 or b <= 1:
            print(f"eprop mseed={ms} lr={lr}: first@{f} hold {h:.2f} "
                  f"best {b}", flush=True)

print("--- (c) BPTT at any e-prop-capable seed will be checked next ---")
