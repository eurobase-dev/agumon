"""Merge the v4 LoRA adapter into the pinned base weights without peft (version-independent).
W <- W + (alpha/r) * B @ A, computed in fp32, stored bf16. usage: merge_lora.py <base_dir> <adapter_dir> <out_dir>"""
import json, os, re, shutil, sys, torch
from safetensors.torch import load_file, save_file
base, adapter, out = sys.argv[1:4]
cfg = json.load(open(f'{adapter}/adapter_config.json')); scale = cfg['lora_alpha'] / cfg['r']
A = load_file(f'{adapter}/adapter_model.safetensors')
pairs = {}
for k, v in A.items():
    m = re.match(r'base_model\.model\.(.+)\.lora_([AB])\.weight$', k)
    assert m, f'unexpected adapter key {k}'
    pairs.setdefault(m.group(1) + '.weight', {})[m.group(2)] = v
os.makedirs(out, exist_ok=True)
for f in os.listdir(base):
    if not f.endswith('.safetensors') and os.path.isfile(f'{base}/{f}'): shutil.copy(f'{base}/{f}', out)
done = 0
for f in sorted(x for x in os.listdir(base) if x.endswith('.safetensors')):
    W = load_file(f'{base}/{f}')
    for name in list(W):
        if name in pairs:
            ab = pairs[name]; W[name] = (W[name].float() + scale * (ab['B'].float() @ ab['A'].float())).to(W[name].dtype); done += 1
    save_file(W, f'{out}/{f}', metadata={'format': 'pt'})
assert done == len(pairs) == 112, f'merged {done} of {len(pairs)} LoRA modules (expected 16 layers x 7 = 112)'
print('merged', done, 'modules, scale', scale)
