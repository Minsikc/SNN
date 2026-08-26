import json, csv
import numpy as np

r = json.load(open('results/xor/xor_analog_res_seed0.json'))
out = np.array(r['best_outputs'])
labels = [0, 1, 1, 0]
names = ['A0B0', 'A0B1', 'A1B0', 'A1B1']
for i in range(4):
    g0 = out[i, 14:20, 0:2].sum()
    g1 = out[i, 14:20, 3:5].sum()
    pred = 1 if g1 > g0 else (0 if g0 > g1 else -1)
    ok = 'OK' if pred == labels[i] else 'WRONG'
    print(f'{names[i]}: g0={g0:.0f} g1={g1:.0f} pred={pred} '
          f'label={labels[i]} {ok}')
print('acc trace:', r['accs'])

rows = list(csv.DictReader(open('results/xor/grad_log_analog.csv')))
adc = np.array([abs(float(x['hw_adc'])) for x in rows])
print(f'|hw_adc| mean {adc.mean():.1f}  p95 {np.percentile(adc, 95):.1f}  '
      f'max {adc.max():.1f}')
