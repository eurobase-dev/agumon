"""Public HTTP benchmarks -> one JSONL per set: {method, url (path+query), headers, data, label block|allow, cls}
and the matching v4 prompts (<name>.prompts.jsonl, prompt + completion) rendered with the v4 formatter.
Sets: CSIC 2010 test (normalTrafficTest = allow, anomalousTrafficTest = block), ECML/PKDD 2007 test (class Valid = allow),
HttpParams (parameter values only; each value is placed as one query parameter of a fixed GET request and as one
form field of a fixed POST, so the model sees a whole request; this wrapping is ours, stated in the paper).
usage: convert.py <v2fmt_dir> <out_dir>   (run from benchmarks/)"""
import csv, json, os, re, sys, urllib.parse, collections
fmt, out = sys.argv[1:3]
sys.path.insert(0, fmt)
from v2fmt import prompt
os.makedirs(out, exist_ok=True)
D = 'Web-Application-Attack-Datasets/OriginalDataSets/'
REQ = re.compile(r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|PROPFIND|[A-Z]{3,10}) (\S+) HTTP/\d\.\d$')
def parse(lines):
    m = REQ.match(lines[0]); method, target = m.group(1), m.group(2)
    u = urllib.parse.urlsplit(target); url = (u.path or '/') + ('?' + u.query if u.query else '') if u.scheme else target
    headers, i = {}, 1
    while i < len(lines) and lines[i] != '':
        k, _, v = lines[i].partition(':'); headers[k.strip()] = v.strip(); i += 1
    body = '\n'.join(l for l in lines[i + 1:]).strip('\n')
    return method, url, headers, body
def csic(path, label):
    raw = open(path, encoding='latin-1').read().split('\n'); cur = []
    for l in raw + ['GET x HTTP/1.1']:
        if REQ.match(l) and cur:
            while cur and cur[-1] == '': cur.pop()
            yield (*parse(cur), label, 'csic_' + label)
            cur = []
        cur.append(l)
def ecml(path):
    block, cls = [], None
    for l in open(path, encoding='latin-1').read().split('\n'):
        if l.startswith('Start - Id:'): block, cls = [], None; continue
        if l.startswith('class:'): cls = l.split(':', 1)[1].strip(); continue
        if l.startswith('End - Id:'):
            while block and block[-1] == '': block.pop()
            if block and REQ.match(block[0]): yield (*parse(block), 'allow' if cls == 'Valid' else 'block', cls)
            continue
        block.append(l)
def httpparams(path):
    H = {'Host': 'shop.example', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0', 'Accept': '*/*'}
    for r in csv.DictReader(open(path, encoding='utf-8', errors='replace')):
        lab = 'allow' if r['label'] == 'norm' else 'block'; v = urllib.parse.quote_plus(r['payload'])
        yield 'GET', '/search?q=' + v, H, '', lab, r['attack_type']
        yield 'POST', '/account', {**H, 'Content-Type': 'application/x-www-form-urlencoded'}, 'name=' + v, lab, r['attack_type']
sets = {'csic2010': [csic(D + 'csic_2010/normalTrafficTest.txt', 'allow'), csic(D + 'csic_2010/anomalousTrafficTest.txt', 'block')],
        'ecml2007': [ecml(D + 'ecml_pkdd/dataset_ecml_pkdd_train_test/xml_test.txt')],
        'httpparams': [httpparams('HttpParamsDataset/payload_test.csv')]}
for name, gens in sets.items():
    st = collections.Counter()
    with open(f'{out}/{name}.jsonl', 'w') as f, open(f'{out}/{name}.prompts.jsonl', 'w') as g:
        for gen in gens:
            for method, url, headers, body, label, cls in gen:
                f.write(json.dumps({'method': method, 'url': url, 'headers': headers, 'data': body, 'label': label, 'cls': cls}) + '\n')
                g.write(json.dumps({'prompt': prompt(method, url, headers, body), 'completion': ' ' + label}) + '\n')
                st[f'{label}'] += 1; st[f'cls:{cls}'] += 1
    print(name, json.dumps(dict(sorted(st.items()))))
