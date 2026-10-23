"""The v4 model as an HTTP endpoint for GoTestWAF: every request (any method, path, headers, body) is scored as received;
403 if p(block) >= frozen v4 threshold, else 200. One model, one thread (MLX streams are per thread): requests are served one at a time.
Logs {method, path_len, p, status} per request. usage: v4_server.py <base> <adapter> <fmt_dir> <log.jsonl> [port=8099]"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from v4_scorer import V4, T
base, adapter, fmt, logp = sys.argv[1:5]; PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 8099
m = V4(base, adapter, fmt); lock = threading.Lock(); log = open(logp, 'a')
class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def handle_any(self):
        n = int(self.headers.get('Content-Length') or 0); body = self.rfile.read(n).decode('utf-8', 'replace') if n else ''
        headers = {k: v for k, v in self.headers.items()}
        with lock:
            p = m.score([(self.command, self.path, headers, body)])[0]
            st = 403 if p >= T else 200
            log.write(json.dumps({'method': self.command, 'path_len': len(self.path), 'p': p, 'status': st}) + '\n'); log.flush()
        out = b'blocked' if st == 403 else b'ok'
        self.send_response(st); self.send_header('Content-Type', 'text/plain'); self.send_header('Content-Length', str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass
for verb in ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'):
    setattr(H, 'do_' + verb, H.handle_any)
print(f'v4 WAF on :{PORT} threshold {T}', flush=True)
HTTPServer(('0.0.0.0', PORT), H).serve_forever()
