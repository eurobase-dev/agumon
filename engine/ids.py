"""Token ids of the 300 validation prompts, built once so every engine times exactly the same input.
usage: ids.py <merged_dir> <prompts.jsonl> <out.jsonl>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from common import Fmt
m, src, out = sys.argv[1:4]; fmt = Fmt(AutoTokenizer.from_pretrained(m))
with open(out, 'w') as f:
    for i, l in enumerate(open(src)):
        r = json.loads(l); f.write(json.dumps({'i': i, 'ids': fmt.ids(r['prompt']), 'label': r['completion'].strip()}) + '\n')
json.dump({'B': fmt.B, 'A': fmt.A}, open(out + '.tokens.json', 'w')); print('ids', i + 1, 'B', fmt.B, 'A', fmt.A)
