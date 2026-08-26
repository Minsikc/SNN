"""e-prop on T=12/seed11: does it reach a perfect raster for model seeds
0/1/2 (the BPTT-perfect seeds are 1 and 2)?"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from sweep_teacher_perfect import build_task, raster_err
from models.models import Basic_RSNN_eprop_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time

T, DS_SEED, EPOCHS = 12, 11, 300

for ms in [0, 1, 2]:
    for lr in [0.05, 0.1, 0.2]:
        x, tgt = build_task(T, DS_SEED)
        torch.manual_seed(ms)
        m = Basic_RSNN_eprop_forward(n_in=10, n_hidden=5, n_out=5,
                                     recurrent=True, init_thresh=0.2)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        kernel = create_exponential_kernel(3, 2.0)
        first, flags = -1, []
        for ep in range(EPOCHS):
            opt.zero_grad()
            out = m(x, tgt, training=True)
            co = apply_convolution(out, kernel, 3)
            ct = apply_convolution(tgt, kernel, 3)
            mse_acc_loss_over_time(co, ct, out.shape[1])
            opt.step()
            with torch.no_grad():
                o = m(x, tgt, training=False)
            missed, extra = raster_err(o, tgt)
            flags.append(missed + extra == 0)
            if flags[-1] and first < 0:
                first = ep + 1
        hold = float(np.mean(flags[-50:]))
        best = min(sum(raster_err((m(x, tgt, training=False) > .5).float(),
                                  tgt)) for _ in [0])
        print(f"mseed={ms} lr={lr}: first@{first} hold(last50) {hold:.2f}",
              flush=True)
