"""Shared tokenization for training and scoring: identical to the mlx training sequence
(chat template with the empty think block '<think>\n\n</think>\n\n', then the decision token)."""
import json
def load_jsonl(path): return [json.loads(l) for l in open(path) if l.strip()]
class Fmt:
    def __init__(self, tok):
        self.tok = tok
        b, a = tok.encode(' block', add_special_tokens=False), tok.encode(' allow', add_special_tokens=False)
        assert len(b) == 1 and len(a) == 1, (b, a)
        self.B, self.A = b[0], a[0]
    def ids(self, prompt):
        # enable_thinking=False makes the template emit '<think>\n\n</think>\n\n', exactly what mlx-lm trained on
        # (assistant turn without reasoning_content); the old '+ </think>\n\n' suffix dropped '<think>\n\n'.
        return list(self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                                 enable_thinking=False, return_dict=False))
    def target(self, completion):
        c = completion.strip(); assert c in ('block', 'allow'), completion
        return self.B if c == 'block' else self.A
