"""Verify the threshold-aligned VRD=0 recipe in the LOCAL codebase.

Mirrors the 2026-08-19 re-sweep from the reference workspace (winning recipe:
threshold 0.5 aligned everywhere, e-prop + Adamax lr 0.01, BPTT, decoupled
seeds, 500 epochs, VRD tau=5) at two scales:

  A) their scale:  100 -> 40 -> 10, T=50, input density 0.1
  B) HW scale   :   10 ->  5 ->  5, T in {12, 25, 50}  (5x5-array constraint)

Per config we also run the sanity check that proves alignment end-to-end:
planting the teacher's own weights into the student must give VRD = 0.

Success criterion per run: exact raster match (missed+extra == 0), which for
an exponential kernel is equivalent to VRD == 0.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.customdatasets import CustomSpikeDataset_Teacher
from models.models import Basic_RSNN_eprop_forward
from utils.metrics import van_rossum_distance

THRESH = 0.5
TAU = 0.6
EPOCHS = 500


def smooth(x, a=0.8):
    B, T, C = x.shape
    f = torch.zeros(B, C)
    out = []
    for t in range(T):
        f = a * f + x[:, t, :]
        out.append(f)
    return torch.stack(out, 1)


def build_task(n_in, H, n_out, T, ds_seed, density=0.1):
    """Teacher task with student-matched threshold/tau. Auto-scales teacher
    weights until the target is non-degenerate (density in [0.03, 0.4])."""
    for w_scale in [1.0, 1.5, 2.0, 2.5]:
        ds = CustomSpikeDataset_Teacher(
            num_samples=1, sequence_length=T, input_size=n_in,
            output_size=n_out, hidden_size=H, spike_prob=density,
            teacher_thresh=THRESH, teacher_tau=TAU, w_scale=w_scale,
            seed=ds_seed)
        d = float(ds.targets.mean())
        if 0.03 <= d <= 0.4:
            return ds, w_scale, d
    return ds, w_scale, d   # last attempt even if degenerate


def build_student(n_in, H, n_out, student_seed):
    torch.manual_seed(student_seed)
    return Basic_RSNN_eprop_forward(
        n_in=n_in, n_hidden=H, n_out=n_out, recurrent=True,
        init_tau=TAU, init_thresh=THRESH, init_tau_o=TAU, gamma=0.3)


def plant_teacher(model, ds):
    """Copy teacher weights into the student (transposing to Linear layout)."""
    tw = ds.teacher_weights
    with torch.no_grad():
        model.fc1.weight.copy_(tw['w_in'].t())
        model.recurrent.copy_(tw['w_rec'])
        model.out.weight.copy_(tw['w_out'].t())


def raster_err(out, tgt):
    o = (out > 0.5).float()
    return int(((tgt == 1) & (o == 0)).sum() + ((tgt == 0) & (o == 1)).sum())


def vrd(out, tgt):
    return float(van_rossum_distance((out > 0.5).float(), tgt, tau=5.0).mean())


def train(rule, ds, model, lr, epochs=EPOCHS):
    x, y = ds.data, ds.targets
    if rule == "bptt":
        model.custom_grad = False
        model.custom_grad_forward = False
        model.LIF0.surrogate_function.subthresh = torch.tensor(0.5)
        model.out_node.surrogate_function.subthresh = torch.tensor(0.5)
    opt = torch.optim.Adamax(model.parameters(), lr=lr)

    first, best_vrd, flags = -1, float("inf"), []
    for ep in range(epochs):
        opt.zero_grad()
        out = model(x, y, training=True)
        if rule == "bptt":
            loss = ((smooth(out) - smooth(y)) ** 2).mean()
            model.init_net()          # pure autograd (clear e-prop grads)
            loss.backward()
        opt.step()
        with torch.no_grad():
            o = model(x, y, training=False)
        err = raster_err(o, y)
        v = vrd(o, y)
        best_vrd = min(best_vrd, v)
        flags.append(err == 0)
        if err == 0 and first < 0:
            first = ep + 1
    hold = float(np.mean(flags[-50:]))
    final_zero = bool(flags[-1])
    return first, hold, best_vrd, final_zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="results/vrd0_verify.json")
    args = ap.parse_args()

    scales = [
        ("A(theirs)", dict(n_in=100, H=40, n_out=10), [50]),
        ("B(HW 5x5)", dict(n_in=10, H=5, n_out=5), [12, 25, 50]),
    ]
    ds_seeds = [21, 22, 23]
    rules = [("eprop", 0.01), ("bptt", 0.01)]

    results = []
    print(f"{'scale':>10} {'T':>4} {'rule':>6} {'seed':>4} | "
          f"{'plant VRD':>9} | {'first0':>6} {'hold':>5} {'bestVRD':>8} "
          f"{'final0':>6}")
    for scale_name, dims, Ts in scales:
        for T in Ts:
            for ds_seed in ds_seeds:
                ds, w_scale, dens = build_task(T=T, ds_seed=ds_seed, **dims)
                # sanity: teacher weights are an exact optimum
                m0 = build_student(student_seed=ds_seed + 100, **dims)
                plant_teacher(m0, ds)
                with torch.no_grad():
                    o0 = m0(ds.data, ds.targets, training=False)
                plant_v = vrd(o0, ds.targets)

                for rule, lr in rules:
                    m = build_student(student_seed=ds_seed + 100, **dims)
                    first, hold, best_v, fz = train(rule, ds, m, lr,
                                                    args.epochs)
                    results.append(dict(
                        scale=scale_name, T=T, ds_seed=ds_seed, rule=rule,
                        lr=lr, w_scale=w_scale, density=dens,
                        plant_vrd=plant_v, first_zero=first, hold=hold,
                        best_vrd=best_v, final_zero=fz))
                    print(f"{scale_name:>10} {T:>4} {rule:>6} {ds_seed:>4} | "
                          f"{plant_v:>9.3f} | {first:>6} {hold:>5.2f} "
                          f"{best_v:>8.3f} {str(fz):>6}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)

    n_zero = sum(r["first_zero"] > 0 for r in results)
    print(f"\nVRD=0 reached in {n_zero}/{len(results)} runs "
          f"(plant sanity: max plant_vrd = "
          f"{max(r['plant_vrd'] for r in results):.4f}, must be 0)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
