"""vLLM single-request latency, batch 1, 1 output token, on the shared ids; 3 timed passes after a warm pass.
mode 'asis' = the configuration of the published 11.6 ms run (engine in a separate process, CUDA graphs up to 512 tokens,
20 logprobs over the vocabulary) except prefix caching, which is off in both modes so repeated passes are not cache hits
(the published run reused only the ~25-token constant prefix); mode 'tuned' = engine in-process, CUDA graphs up to 2048 tokens, only the two decision
logits. Writes one JSON line per request and pass: mode, pass, i, len, wall_ms, margin (logit block - logit allow).
usage: lat_vllm.py <merged_dir> <ids.jsonl> <asis|tuned> <out.jsonl>"""
import json, os, sys, time
def main():
    m, data, mode, out = sys.argv[1:5]
    if mode == 'tuned': os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    from vllm import LLM, SamplingParams, TokensPrompt
    rows = [json.loads(l) for l in open(data)]; tk = json.load(open(data + '.tokens.json')); B, A = tk['B'], tk['A']
    kw = dict(model=m, dtype='bfloat16', max_model_len=8192, enable_prefix_caching=False, gpu_memory_utilization=0.6)
    if mode == 'tuned':
        kw.update(logprobs_mode='raw_logits', compilation_config={'max_cudagraph_capture_size': 2048})
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=[B, A])
    else:
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    llm = LLM(**kw)
    o = llm.generate([TokensPrompt(prompt_token_ids=rows[0]['ids'])], sp, use_tqdm=False)[0]
    print('FIRST_LOGPROBS', {k: round(v.logprob, 4) for k, v in o.outputs[0].logprobs[0].items()}, flush=True)
    def one(x):
        t = time.perf_counter(); o = llm.generate([TokensPrompt(prompt_token_ids=x)], sp, use_tqdm=False)[0]
        ms = (time.perf_counter() - t) * 1000; lp = o.outputs[0].logprobs[0]
        mg = lp[B].logprob - lp[A].logprob if B in lp and A in lp else None   # None: a decision token fell outside the top 20
        return ms, mg
    with open(out, 'w') as f:
        for p in range(4):   # pass 0 warms every shape; prefix caching is off so repeated passes are not cache hits
            for r in rows:
                ms, mg = one(r['ids'])
                if p: f.write(json.dumps({'engine': 'vllm_' + mode, 'pass': p, 'i': r['i'], 'len': len(r['ids']), 'wall_ms': round(ms, 3), 'margin': mg}) + '\n')
    print('VLLM_DONE', mode, flush=True)
if __name__ == '__main__': main()
