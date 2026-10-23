"""Minimal dedicated engine: one process, merged model on the GPU, batch 1, only the two decision logits at the last
prompt position. Variants: bf16 eager (reference), bf16 and FP8 (torchao dynamic per-row) with torch.compile
reduce-overhead (CUDA graphs), prompts right-padded to multiples of 64 (causal model, readout at the real last token).
Per request and pass: wall_ms (host ids -> margin on host); for the graph variants also gpu_ms (CUDA events around the
replay; in eager the same span would include host launch gaps, so it is not recorded there).
Prints DIRECT_FALLBACK with the counters if any CUDA graph is skipped or a graph break occurs, so an eager fallback is visible.
usage: lat_direct.py <merged_dir> <ids.jsonl> <out.jsonl>"""
import json, sys, time, torch
import torch._dynamo, torch._inductor.config
from torch._dynamo.utils import counters
for k in ('recompile_limit', 'cache_size_limit', 'accumulated_recompile_limit', 'accumulated_cache_size_limit'):
    if hasattr(torch._dynamo.config, k): setattr(torch._dynamo.config, k, 512)
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
from transformers import AutoModelForCausalLM
m, data, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(data)]; tk = json.load(open(data + '.tokens.json'))
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
W2 = model.lm_head.weight[[tk['B'], tk['A']]].detach().clone()
def fwd(x, last):
    h = model.model(input_ids=x, use_cache=False).last_hidden_state
    lg = (h[0].index_select(0, last) @ W2.T).float()[0]
    return lg[0] - lg[1]
f_out = open(out, 'w')
def run(name, f, pad, passes):
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    W = 2 if pad else 1   # CUDA graph trees warm a new shape on its first call and record it on the second: skip both
    for p in range(passes + W):
        for r in rows:
            x = r['ids']; L = -(-len(x) // 64) * 64 if pad else len(x)
            torch.cuda.synchronize(); t = time.perf_counter()
            xt = torch.tensor([x + [0] * (L - len(x))], device='cuda'); last = torch.tensor([len(x) - 1], device='cuda')
            with torch.inference_mode():
                ev0.record(); mg = f(xt, last); ev1.record(); mg = mg.item()
            wall = (time.perf_counter() - t) * 1000; gpu = ev0.elapsed_time(ev1)
            if p >= W: f_out.write(json.dumps({'engine': name, 'pass': p - W + 1, 'i': r['i'], 'len': len(x), 'wall_ms': round(wall, 3), **({'gpu_ms': round(gpu, 3)} if pad else {}), 'margin': mg}) + '\n')
    f_out.flush()
    skips, breaks = sum(counters['inductor'].get(k, 0) for k in counters['inductor'] if 'cudagraph_skip' in k), sum(counters['graph_break'].values())
    print('DIRECT', name, 'cudagraph_skips', skips, 'graph_breaks', breaks, 'unique_graphs', counters['stats'].get('unique_graphs', 0), flush=True)
    if pad and (skips or breaks): print('DIRECT_FALLBACK', name, dict(counters['inductor']), dict(counters['graph_break']), flush=True)
run('bf16_eager', fwd, False, 1)
run('bf16_graph', torch.compile(fwd, mode='reduce-overhead', dynamic=True), True, 3)
torch._dynamo.reset(); counters.clear()
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow
quantize_(model.model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
run('fp8_graph', torch.compile(fwd, mode='reduce-overhead', dynamic=True), True, 3)
print('DIRECT_DONE', flush=True)
