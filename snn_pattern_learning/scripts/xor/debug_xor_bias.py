"""Per-cell systematic bias analysis of the analog gradient log.

If the crossbar gradient has a uniform (or cell-specific) offset relative to
the desired gradient, averaging over repeats cannot remove it -- that is the
signature that distinguishes 'noise' from 'bias'.
"""
import csv
import numpy as np

rows = list(csv.DictReader(open('results/xor/grad_log_analog.csv')))
eps = sorted({int(r['epoch']) for r in rows})
D = np.zeros((len(eps), 5, 5))
H = np.zeros((len(eps), 5, 5))
for r in rows:
    e = eps.index(int(r['epoch']))
    D[e, int(r['row']) - 1, int(r['col']) - 1] = float(r['desired'])
    H[e, int(r['row']) - 1, int(r['col']) - 1] = float(r['hw_scaled'])

diff = H - D
print(f"epochs: {len(eps)}")
print(f"global bias  mean(hw - desired) = {diff.mean():+.4f}  "
      f"(|desired| mean {np.abs(D).mean():.4f})")
print(f"per-epoch bias mean: {diff.mean(axis=(1, 2)).round(3)}")

print("\nper-cell mean(hw - desired)  [rows = out neuron, cols = hidden]:")
print(diff.mean(axis=0).round(3))

print("\nper-cell corr(desired, hw) over epochs:")
C = np.full((5, 5), np.nan)
for i in range(5):
    for j in range(5):
        d, h = D[:, i, j], H[:, i, j]
        if d.std() > 0 and h.std() > 0:
            C[i, j] = np.corrcoef(d, h)[0, 1]
print(np.round(C, 2))

print("\nper-cell mean desired vs mean hw:")
print("desired:\n", D.mean(axis=0).round(3))
print("hw:\n", H.mean(axis=0).round(3))
