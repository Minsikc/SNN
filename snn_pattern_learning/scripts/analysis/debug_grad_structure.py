"""Compare desired-gradient STRUCTURE: teacher (worked on HW) vs XOR (stuck).

Metrics per log:
  scale    mean |desired| per epoch
  conc     column concentration: share of total |desired| carried by the
           largest column (0.2 = perfectly even across 5 columns)
  mixed    fraction of significant columns whose rows carry BOTH signs
           (each sign >=20% of the column's max magnitude)
  persist  per-cell |mean over epochs| / mean|.| over epochs, averaged over
           significant cells (1.0 = the same demand every epoch;
           low = demands fluctuate/cancel across epochs)
  hwr      pooled corr(desired, hw_scaled) for reference
"""
import csv
import numpy as np

LOGS = [
    ("teacher seed0", "results/eprop_grad_log/gradient_log_seed0.csv"),
    ("teacher seed1", "results/eprop_grad_log/gradient_log_seed1.csv"),
    ("teacher n5T20", "results/eprop_grad_log/grad_n5_T20.csv"),
    ("XOR run1 (K=1)", "results/xor/grad_log_analog_run1.csv"),
    ("XOR run2 (K=3)", "results/xor/grad_log_analog_run2.csv"),
    ("XOR run3 (batch)", "results/xor/grad_log_analog.csv"),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    eps = sorted({int(r["epoch"]) for r in rows})
    D = np.zeros((len(eps), 5, 5))
    H = np.zeros((len(eps), 5, 5))
    for r in rows:
        e = eps.index(int(r["epoch"]))
        D[e, int(r["row"]) - 1, int(r["col"]) - 1] = float(r["desired"])
        H[e, int(r["row"]) - 1, int(r["col"]) - 1] = float(r["hw_scaled"])
    return D, H


print(f"{'log':>18} {'epochs':>6} {'scale':>7} {'conc':>5} {'mixed':>6} "
      f"{'persist':>7} {'hw r':>6}")
for name, path in LOGS:
    try:
        D, H = load(path)
    except FileNotFoundError:
        print(f"{name:>18}  (missing)")
        continue
    E = len(D)
    scale = np.abs(D).mean()

    col_mag = np.abs(D).sum(axis=1)          # (E, 5) column magnitudes
    tot = col_mag.sum(axis=1, keepdims=True) + 1e-12
    conc = (col_mag / tot).max(axis=1).mean()

    mixed_fracs = []
    for e in range(E):
        cols_mixed, cols_sig = 0, 0
        for j in range(5):
            c = D[e, :, j]
            m = np.abs(c).max()
            if m < 1e-6:
                continue
            cols_sig += 1
            pos = (c > 0.2 * m).any()
            neg = (c < -0.2 * m).any()
            if pos and neg:
                cols_mixed += 1
        if cols_sig:
            mixed_fracs.append(cols_mixed / cols_sig)
    mixed = np.mean(mixed_fracs) if mixed_fracs else float("nan")

    mean_abs = np.abs(D).mean(axis=0)        # (5,5)
    abs_mean = np.abs(D.mean(axis=0))
    sig = mean_abs > 0.1 * mean_abs.max()
    persist = (abs_mean[sig] / (mean_abs[sig] + 1e-12)).mean()

    d, h = D.ravel(), H.ravel()
    hwr = np.corrcoef(d, h)[0, 1] if d.std() > 0 and h.std() > 0 else np.nan

    print(f"{name:>18} {E:>6} {scale:>7.3f} {conc:>5.2f} {mixed:>6.2f} "
          f"{persist:>7.2f} {hwr:>6.2f}")
