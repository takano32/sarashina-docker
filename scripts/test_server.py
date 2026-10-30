#!/usr/bin/env python3
"""server.py を偽の上流に向けて確かめる。鍵も通信も要らない: python3 scripts/test_server.py"""
import base64, json, math, os, struct, subprocess, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

seen, fail_once = [], {"429": True}

class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["input"] == "rate" and fail_once["429"]:
            fail_once["429"] = False
            self.send_response(429); self.send_header("Retry-After", "1"); self.send_header("Content-Length", "0"); self.end_headers(); return
        if body["input"] == "bad":
            data = b'{"error":"too long"}'
            self.send_response(400); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        seen.append((body["input"], body["task_type"], self.headers["Authorization"]))
        data = json.dumps({"data": [{"embedding": [3.0, 4.0]}], "usage": {"prompt_tokens": 2}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass

upstream = HTTPServer(("127.0.0.1", 18091), Fake)
threading.Thread(target=upstream.serve_forever, daemon=True).start()
server_py = Path(__file__).resolve().parent.parent / "server.py"
env = dict(os.environ, SARASHINA_BASE_URL="http://127.0.0.1:18091", SARASHINA_API_KEY="dummy", PORT="18092")
proc = subprocess.Popen([sys.executable, str(server_py)], env=env, stderr=subprocess.DEVNULL)

def post(payload):
    req = urllib.request.Request("http://127.0.0.1:18092/v1/embeddings", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

try:
    for _ in range(50):
        try:
            urllib.request.urlopen("http://127.0.0.1:18092/health", timeout=1); break
        except OSError:
            time.sleep(0.1)
    # 配列を分けて印を task_type に振り替え、印は上流に送らない
    status, r = post({"input": ["[[sarashina:query]]問い", "[[sarashina:document]]文書", "印なし"], "model": "x"})
    assert status == 200 and len(r["data"]) == 3 and r["usage"]["total_tokens"] == 6, r
    assert seen == [("問い", "RETRIEVAL_QUERY", "Bearer dummy"), ("文書", "RETRIEVAL_DOCUMENT", "Bearer dummy"), ("印なし", "RETRIEVAL_DOCUMENT", "Bearer dummy")], seen
    # 正規化
    v = r["data"][0]["embedding"]; assert math.isclose(v[0], 0.6) and math.isclose(v[1], 0.8), v
    # 文字列の input と base64
    status, r = post({"input": "単発", "encoding_format": "base64"})
    assert status == 200
    decoded = struct.unpack("<2f", base64.b64decode(r["data"][0]["embedding"]))
    assert math.isclose(decoded[0], 0.6, rel_tol=1e-6), decoded
    # 429 は待って送り直す
    t = time.monotonic(); status, r = post({"input": "rate"})
    assert status == 200 and time.monotonic() - t >= 1, (status, r)
    # 上流の 400 はそのまま返す
    status, r = post({"input": "bad"}); assert status == 400, r
    # トークン列は断る
    status, r = post({"input": [[1, 2]]}); assert status == 400, r
    print("ok")
finally:
    proc.terminate()
