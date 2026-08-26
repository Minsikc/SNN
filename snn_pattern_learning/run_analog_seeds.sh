#!/bin/bash
# Analog runs for several seeds. Each needs its own config because init_seed
# is read from the yaml, and its own gradient log so the per-epoch fidelity
# can be attributed to the right seed.
for SD in 0 1 2; do
  echo "########## analog seed $SD ##########"
  python - "$SD" <<'PY'
import io,sys
sd=sys.argv[1]
s=io.open('configs/eprop_grad_log.yaml',encoding='utf-8').read()
import re
s=re.sub(r'  init_seed: \d+', f'  init_seed: {sd}', s)
s=re.sub(r'grad_log_path: "[^"]*"',
         f'grad_log_path: "results/eprop_grad_log/gradient_log_seed{sd}.csv"', s)
s=re.sub(r'  name: "[^"]*"', f'  name: "eprop_seed{sd}"', s)
io.open(f'configs/eprop_seed{sd}.yaml','w',encoding='utf-8').write(s)
print('config written for seed', sd)
PY
  rm -f "results/eprop_grad_log/gradient_log_seed${SD}.csv"
  python -u main_unified.py --config "eprop_seed${SD}.yaml" --epochs 50 2>&1 \
    | grep -E "^\[INIT\]|^Epoch 1/|^Epoch 50/|Results:"
done
