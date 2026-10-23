"""Convert a PEFT LoRA adapter (from the GPU run) back to mlx-lm format. usage: peft2mlx.py <peft_dir> <mlx_template_cfg> <out_dir>"""
import json, sys, os
from safetensors.numpy import load_file, save_file
src, tmpl, dst = sys.argv[1:4]; os.makedirs(dst, exist_ok=True)
w = load_file(f'{src}/adapter_model.safetensors'); out = {}
for k, v in w.items():                      # base_model.model.model.layers.N.mlp.up_proj.lora_A.weight
    base = k.replace('base_model.model.', '', 1).rsplit('.', 2)[0]
    kind = 'lora_a' if '.lora_A.' in k else 'lora_b'
    out[f'{base}.{kind}'] = v.T.copy()
save_file(out, f'{dst}/adapters.safetensors')
cfg = json.load(open(tmpl)); cfg['adapter_path'] = dst; json.dump(cfg, open(f'{dst}/adapter_config.json', 'w'), indent=2)
print(len(out), 'tensors ->', dst)
