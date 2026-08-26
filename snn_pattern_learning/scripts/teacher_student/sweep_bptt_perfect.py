"""Focused BPTT tuning on the task where e-prop already reaches a perfect
raster (T=12, dataset seed 11): surrogate width x lr x model seed.
Goal: pure-BPTT exact raster (missed+extra = 0), held.
"""
import os
import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from sweep_teacher_perfect import build_task, raster_err
from models.models import Basic_RSNN_eprop_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time

T, DS_SEED, EPOCHS = 12, 11, 1000


def run_bptt(lr, subthresh, model_seed):
    x, tgt = build_task(T, DS_SEED)
    torch.manual_seed(model_seed)
    m = Basic_RSNN_eprop_forward(n_in=10, n_hidden=5, n_out=5,
                                 recurrent=True, init_thresh=0.2)
    m.custom_grad = False
    m.custom_grad_forward = False
    m.LIF0.surrogate_function.subthresh = torch.tensor(subthresh)
    m.out_node.surrogate_function.subthresh = torch.tensor(subthresh)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    kernel = create_exponential_kernel(3, 2.0)

    first, flags, best = -1, [], 99
    for ep in range(EPOCHS):
        opt.zero_grad()
        out = m(x, tgt, training=True)
        co = apply_convolution(out, kernel, 3)
        ct = apply_convolution(tgt, kernel, 3)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1])
        m.init_net()
        loss.backward()
        opt.step()
        with torch.no_grad():
            o = m(x, tgt, training=False)
        missed, extra = raster_err(o, tgt)
        err = missed + extra
        best = min(best, err)
        flags.append(err == 0)
        if err == 0 and first < 0:
            first = ep + 1
    hold = float(np.mean(flags[-100:]))
    return first, hold, best


rows = []
for subthresh, lr, ms in itertools.product(
        [0.1, 0.25, 0.5], [0.01, 0.03, 0.1], [0, 1, 2]):
    first, hold, best = run_bptt(lr, subthresh, ms)
    rows.append((first, hold, best, subthresh, lr, ms))
    print(f"subthr={subthresh} lr={lr} mseed={ms}: "
          f"first@{first} hold(last100) {hold:.2f} best {best}", flush=True)

ok = [r for r in rows if r[0] > 0]
ok.sort(key=lambda r: (-r[1], r[0]))
print("\n=== perfect configs ===")
for first, hold, best, st, lr, ms in ok[:8]:
    print(f"subthr={st} lr={lr} mseed={ms}: first@{first} hold {hold:.2f}")
