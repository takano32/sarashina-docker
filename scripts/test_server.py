#!/usr/bin/env python3
"""server.py を偽の上流に向けて確かめる。鍵も通信も要らない: python3 scripts/test_server.py"""
import base64, json, math, os, struct, subprocess, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

seen, chats, fail_once = [], [], {"429": True}

class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/rerank":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = len(body["documents"]); assert n <= 20
            data = json.dumps({"data": [{"result": [{"document_id": i, "score": (i + 1) / 100} for i in reversed(range(n))]}], "usage": {"prompt_tokens": n}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if self.path == "/v1/translate":
            self.rfile.read(int(self.headers["Content-Length"]))
            data = b'{"translated_text": "hello"}'
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if self.path == "/v1/chat/completions":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chats.append(body)
            content = "\u6709\u5bb3\nT4" if body.get("model") == "sarashina3-guard" else "ok"
            data = json.dumps({"id": "c", "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": content}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
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

def post(payload, path="/v1/embeddings"):
    req = urllib.request.Request("http://127.0.0.1:18092" + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
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
    # チャットの読み替え
    status, r = post({"model": "m", "messages": [], "max_tokens": 99999, "logprobs": True, "tools": [{"type": "function"}], "tool_choice": {"type": "function", "function": {"name": "f"}}}, "/v1/chat/completions")
    assert status == 200 and r["choices"][0]["message"]["content"] == "ok", r
    c = chats[-1]; assert c["max_completion_tokens"] == 8000 and "max_tokens" not in c and "logprobs" not in c and c["tool_choice"] == "required", c
    status, r = post({"model": "m", "messages": [], "tools": [{"type": "function"}]}, "/v1/chat/completions")
    assert chats[-1]["tool_choice"] == "auto", chats[-1]
    # rerank は 20 件ずつに分けて並べ直す
    status, r = post({"query": "q", "documents": ["d"] * 25, "top_n": 3}, "/v1/rerank")
    assert status == 200 and len(r["results"]) == 3 and r["results"][0]["relevance_score"] == 0.2 and r["results"][0]["index"] == 19, r
    status, r = post({"query": "q", "texts": ["a", {"text": "b"}], "return_documents": True}, "/rerank")
    assert r["results"][0]["index"] == 1 and r["results"][0]["document"]["text"] == "b", r
    # moderations、completions、translate
    status, r = post({"input": "x"}, "/v1/moderations"); assert r["results"][0]["flagged"] and r["results"][0]["categories"] == {"T4": True}, r
    status, r = post({"prompt": "x", "max_tokens": 5}, "/v1/completions"); assert r["choices"][0]["text"] == "ok" and chats[-1]["max_completion_tokens"] == 5, r
    status, r = post({"text": "x", "source_lang": "ja", "target_lang": "en"}, "/v1/translate"); assert r["translated_text"] == "hello", r
    print("ok")
finally:
    proc.terminate()
