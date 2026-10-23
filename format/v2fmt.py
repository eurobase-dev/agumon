"""Shared request rendering for training and evaluation (same truncation, same prompt)."""
HEAD = 'App: unknown\nRoutes:\n(not declared)\nFiles:\n(not declared)\n\nHTTP request:\n'
Q = '\n\nQuestion: is this request an attack or a vulnerability scan that the platform should block before it reaches the app? Answer allow or block.\nAnswer:'
MAX_URL, MAX_HDR, MAX_BODY = 2000, 1000, 2500
def cap(s, n): return s if len(s) <= n else s[:n] + '…'
def render(method, url, headers, body):
    lines = [f'{method} {cap(url, MAX_URL)} HTTP/1.1'] + [f'{k}: {cap(str(v), MAX_HDR)}' for k, v in headers.items()]
    return '\n'.join(lines) + '\n\n' + cap(body or '', MAX_BODY)
def prompt(method, url, headers, body): return HEAD + render(method, url, headers, body) + Q
