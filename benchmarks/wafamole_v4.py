"""WAF-A-MoLE (Demetrio et al., SAC 2020) against the v4 model. Uses WAF-A-MoLE's own SqlFuzzer (semantics-preserving
MySQL mutations) and reproduces EvasionEngine.evaluate exactly (same pool/priority loop, round size, stop rule), scoring
each round's mutants in one batch. The SQL payload is placed as one query parameter of a fixed GET request (same wrapping
as HttpParams). Initial payloads: SQL injections from the HttpParams test set that v4 blocks, seeded sample.
Evasion = a mutant scored below the frozen v4 threshold. Writes one JSON line per payload.
usage: wafamole_v4.py <base> <adapter> <fmt_dir> <httpparams payload_test.csv> <out.jsonl> [n=40] [max_rounds=500]"""
import csv, json, random, sys, time, urllib.parse
sys.path.insert(0, 'WAF-A-MoLE')
from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer
from v4_scorer import V4, T
base, adapter, fmt, src, out = sys.argv[1:6]; N = int(sys.argv[6]) if len(sys.argv) > 6 else 40; MAXR = int(sys.argv[7]) if len(sys.argv) > 7 else 500
H = {'Host': 'shop.example', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0', 'Accept': '*/*'}
wrap = lambda s: ('GET', '/search?q=' + urllib.parse.quote_plus(s), H, '')
m = V4(base, adapter, fmt)
sqli = [r['payload'] for r in csv.DictReader(open(src, encoding='utf-8', errors='replace')) if r['attack_type'] == 'sqli']
random.Random(20261022).shuffle(sqli)
start = []
for i in range(0, len(sqli), 64):
    chunk = sqli[i:i + 64]; ps = m.score([wrap(s) for s in chunk])
    start += [(s, p) for s, p in zip(chunk, ps) if p >= T]
    if len(start) >= N: break
start = start[:N]
def round_(payload, size=20):
    f = SqlFuzzer(payload); cands = list({f.fuzz() for _ in range(size)}); ps = m.score([wrap(c) for c in cands])
    return min(zip(ps, cands))
random.seed(20261022)
with open(out, 'w') as fo:
    for k, (p0, s0) in enumerate(start):   # p0 = initial payload, s0 = its v4 score
        t0 = time.time(); queries = 0; rounds = MAXR
        conf, pay = round_(p0); queries += 20; res = [(conf, pay)]
        while rounds > 0 and conf > T:
            for cc, cp in sorted(res):
                rounds -= 1; c2, p2 = round_(cp); queries += 20
                if c2 < cc:
                    res.append((c2, p2)); conf, pay = min(res); break
                if rounds <= 0: break
        r = {'i': k, 'start_p': s0, 'min_p': conf, 'evaded': conf < T, 'rounds_used': MAXR - rounds, 'queries': queries,
             'seconds': round(time.time() - t0, 1), 'start_payload': p0, 'final_payload': pay}
        fo.write(json.dumps(r) + '\n'); fo.flush()
        print(f"{k + 1}/{len(start)} evaded={r['evaded']} min_p={conf:.6f} rounds={r['rounds_used']} {r['seconds']}s", flush=True)
