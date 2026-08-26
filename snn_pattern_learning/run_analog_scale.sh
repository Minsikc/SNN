#!/bin/bash
# Analog runs across sequence length and sequence count.
# Each condition gets its own config and gradient log so the per-epoch
# hw_adc values can be checked for saturation (the array clips near
# +455 / -444 LSB, measured 2026-08-08).
run_one () {
  NSEQ=$1; T=$2
  TAG="n${NSEQ}_T${T}"
  echo "########## analog ${TAG} ##########"
  python - "$NSEQ" "$T" "$TAG" <<'PY'
import io,re,sys
nseq,T,tag=sys.argv[1],sys.argv[2],sys.argv[3]
s=io.open('configs/eprop_grad_log.yaml',encoding='utf-8').read()
s=re.sub(r'  name: "[^"]*"', f'  name: "eprop_{tag}"', s)
s=re.sub(r'  num_samples: \d+', f'  num_samples: {nseq}', s)
s=re.sub(r'  sequence_length: \d+', f'  sequence_length: {T}', s)
s=re.sub(r'grad_log_path: "[^"]*"',
         f'grad_log_path: "results/eprop_grad_log/grad_{tag}.csv"', s)
io.open(f'configs/eprop_{tag}.yaml','w',encoding='utf-8').write(s)
print(f'config: n_seq={nseq} T={T}')
PY
  rm -f "results/eprop_grad_log/grad_${TAG}.csv"
  python -u main_unified.py --config "eprop_${TAG}.yaml" --epochs 50 2>&1 \
    | grep -E "^\[INIT\]|^Epoch 1/|^Epoch 50/|Results:"
}
for spec in "1 40" "1 80" "2 20" "5 20"; do
  run_one $spec
done
