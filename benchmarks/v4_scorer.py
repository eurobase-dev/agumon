"""v4 model as a function: list of (method, url, headers, body) -> p(block). MLX, same tokens and LoRA as train_mlx.py,
v4 prompt formatter, frozen v4 threshold. Shared by the GoTestWAF server and the WAF-A-MoLE harness."""
import os, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
T = 0.9986749774553889
class V4:
    def __init__(self, base, adapter, fmt_dir, mem_gb=12):
        sys.path.insert(0, fmt_dir)
        from v2fmt import prompt
        self.prompt = prompt
        mx.set_memory_limit(int(mem_gb * 1024**3)); mx.set_cache_limit(1024**3)
        self.model, self.tok = load(base)
        self.model.freeze()
        linear_to_lora_layers(self.model, 16, {'rank': 8, 'scale': 20.0, 'dropout': 0.0, 'keys': ['self_attn.q_proj', 'self_attn.k_proj',
                              'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']})
        self.model.load_weights(os.path.join(adapter, 'adapters.safetensors'), strict=False)
        self.B = self.tok.encode(' block', add_special_tokens=False)[0]; self.A = self.tok.encode(' allow', add_special_tokens=False)[0]
    def ids(self, method, url, headers, body):
        return self.tok.apply_chat_template([{'role': 'user', 'content': self.prompt(method, url, dict(headers), body or '')}],
                                            add_generation_prompt=True, enable_thinking=False, return_dict=False)
    def score(self, reqs, budget=8192):
        seqs = [self.ids(*r) for r in reqs]; order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); out = [0.0] * len(seqs); i = 0
        while i < len(order):
            L = -(-len(seqs[order[i]]) // 64) * 64; k = 1
            while i + k < len(order) and (-(-len(seqs[order[i + k]]) // 64) * 64) * (k + 1) <= max(budget, L) and k < 64: k += 1
            idx = order[i:i + k]; L = -(-max(len(seqs[j]) for j in idx) // 64) * 64
            x = np.zeros((len(idx), L), dtype=np.int32)
            for r, j in enumerate(idx): x[r, :len(seqs[j])] = seqs[j]
            h = self.model.model(mx.array(x)); last = mx.array([len(seqs[j]) - 1 for j in idx])
            lg = self.model.lm_head(h[mx.arange(len(idx)), last]).astype(mx.float32)
            p = mx.sigmoid(lg[:, self.B] - lg[:, self.A]).tolist()
            for j, v in zip(idx, p): out[j] = v
            i += k
        return out
