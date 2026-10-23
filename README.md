# Agumon

Benchmark harness and inference engines for **Agumon v1.0**, a LoRA adapter for
[MiniCPM5-2B-Base](https://huggingface.co/openbmb/MiniCPM5-2B-Base) that reads one raw HTTP request and decides in a
single forward pass whether a web application firewall should block or allow it.

- Weights, `waf_classify.py` and the frozen threshold: [huggingface.co/giorebil/Agumon-v1.0](https://huggingface.co/giorebil/Agumon-v1.0)
- Writeup: [eurobase.dev/research/agumon](https://eurobase.dev/research/agumon)
- Paper: *A Small Language Model as an Inline Web Application Firewall* (arXiv submission in progress)

These are the scripts that produced the paper's benchmark and latency numbers. Training data, training scripts and the
held-out test set are not included.

## Layout

| Path | What it does |
|---|---|
| `format/v2fmt.py` | Request rendering and truncation the model was trained on (identical to `waf_classify.py`) |
| `benchmarks/peft2mlx.py` | Converts the published PEFT adapter to the mlx-lm format |
| `benchmarks/convert.py` | CSIC 2010, ECML/PKDD 2007 and HttpParams to JSONL requests and prompts |
| `benchmarks/v4_scorer.py` | The model as a function: requests to p(block), MLX, frozen threshold |
| `benchmarks/v4_server.py` | The model as an HTTP endpoint for GoTestWAF: 403 on block, 200 on allow |
| `benchmarks/wafamole_v4.py` | WAF-A-MoLE SQL mutation evasion against the model |
| `engine/merge_lora.py` | Merges the adapter into the pinned base weights, without peft |
| `engine/ids.py`, `engine/common.py` | Token ids built once, so every engine times the same input |
| `engine/lat_direct.py` | bf16 eager reference and torch.compile variants |
| `engine/lat_vllm.py` | vLLM, default and tuned |
| `engine/fast_engine.py` | Minimal hand-written engine, one CUDA graph per 64-token bucket |
| `engine/report.py` | Latency table and decision changes against the bf16 eager reference |

## Setup

```bash
pip install huggingface_hub safetensors numpy
hf download giorebil/Agumon-v1.0 --local-dir agumon-v1.0
hf download openbmb/MiniCPM5-2B-Base --revision 96a57cd572a02506b4500f54427dca24970c1bac --local-dir minicpm5-2b-base
```

The adapter was trained on that exact base revision.

## Benchmarks (Apple silicon, MLX)

```bash
pip install mlx-lm
python benchmarks/peft2mlx.py agumon-v1.0 agumon-v1.0/adapter_config.json agumon-v1.0-mlx
cd benchmarks
git clone https://github.com/msudol/Web-Application-Attack-Datasets
git clone https://github.com/Morzeux/HttpParamsDataset
python convert.py ../format out
```

GoTestWAF scores the model through `v4_server.py`:

```bash
python v4_server.py ../minicpm5-2b-base ../agumon-v1.0-mlx ../format gotestwaf.jsonl 8099
```

then run [GoTestWAF](https://github.com/wallarm/gotestwaf) against `http://localhost:8099`.

WAF-A-MoLE needs its repository next to the script:

```bash
git clone https://github.com/AvalZ/WAF-A-MoLE
python wafamole_v4.py ../minicpm5-2b-base ../agumon-v1.0-mlx ../format HttpParamsDataset/payload_test.csv wafamole.jsonl
```

## Latency (one NVIDIA GPU, CUDA)

```bash
pip install torch transformers vllm
python engine/merge_lora.py minicpm5-2b-base agumon-v1.0 merged
python engine/ids.py merged benchmarks/out/httpparams.prompts.jsonl ids.jsonl
python engine/lat_direct.py merged ids.jsonl direct.jsonl
python engine/fast_engine.py merged ids.jsonl direct.jsonl fast.jsonl bf16
python engine/lat_vllm.py merged ids.jsonl tuned vllm.jsonl
python engine/report.py 0.9986749774553889 direct.jsonl fast.jsonl vllm.jsonl
```

`ids.py` takes any JSONL of `{prompt, completion}` rows. The paper timed 300 validation requests that are not public,
so numbers on other inputs differ with request length.

Versions used: vLLM 0.31.0, transformers 5.17.0 and peft 0.21.2 for latency on one H100 SXM; mlx-lm 0.32 for the
benchmarks on Apple silicon.

## License

Apache License 2.0. See `NOTICE` for the base model and training data attributions.
