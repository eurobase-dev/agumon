"""Table from the per-request JSONL files: per engine the median over passes of p50/p90/p99 wall ms, p50 GPU ms, a linear fit
wall_ms = a + b * tokens (a = fixed cost per request, b = cost per token), and decision changes vs bf16 eager at the frozen threshold.
usage: report.py <threshold> <files...>"""
import json, math, statistics as st, sys
T = float(sys.argv[1]); lt = math.log(T / (1 - T)); rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
eng = {}
for r in rows: eng.setdefault(r['engine'], []).append(r)
ref = {r['i']: r['margin'] for r in eng.get('bf16_eager', [])}
def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
print('| engine | p50 ms | p90 ms | p99 ms | GPU p50 ms | fixed a ms | per 100 tok ms | decisions changed | max |Δmargin| |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for e, rs in eng.items():
    ps = sorted({r['pass'] for r in rs}); by = {p: [r for r in rs if r['pass'] == p] for p in ps}
    med = lambda fn: round(st.median(fn(by[p]) for p in ps), 2)
    p50, p90, p99 = (med(lambda v, k=k: q([r['wall_ms'] for r in v], k)) for k in (.5, .9, .99))
    g = med(lambda v: q([r['gpu_ms'] for r in v], .5)) if 'gpu_ms' in rs[0] else '-'
    x = [r['len'] for r in rs]; y = [r['wall_ms'] for r in rs]; mx, my = st.mean(x), st.mean(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x); a = my - b * mx
    one = by[ps[0]]; miss = sum(r['margin'] is None for r in one)
    ch = sum((r['margin'] >= lt) != (ref[r['i']] >= lt) for r in one if r['margin'] is not None) if ref else '-'
    dm = round(max(abs(r['margin'] - ref[r['i']]) for r in one if r['margin'] is not None), 4) if ref else '-'
    print(f'| {e} | {p50} | {p90} | {p99} | {g} | {round(a, 2)} | {round(100 * b, 2)} | {ch}/{len(one)}{f" ({miss} missing)" if miss else ""} | {dm} |')
n = len(ref); print(f'\n{n} requests, mean {round(st.mean(r["len"] for r in eng["bf16_eager"]))} tokens; threshold {T}')
