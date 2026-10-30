#!/usr/bin/env python3
"""Sarashina の Embeddings API を OpenAI 互換の /v1/embeddings で出す変換サーバー。

違いを埋めるのは次のとおり（理由は README）。
- 配列の input を1件ずつに分けて順に送る（Sarashina は1リクエスト1件）
- 文頭の印を取り除いて task_type に振り替える（Sarashina は task_type が必須）
- ベクトルを長さ1に正規化する（Sarashina の出力は正規化されていない）
- 429 は Retry-After のぶん待って送り直す（60 回/分）
- encoding_format=base64 に応える（OpenAI の SDK の既定。Sarashina は float のみ）

本文はログに出さない。標準ライブラリだけで動く。
"""
import base64
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("SARASHINA_BASE_URL", "https://api.sarashina.ai").rstrip("/")
MODEL = os.environ.get("SARASHINA_EMBED_MODEL", "sarashina3-embedding")
QUERY_MARK = os.environ.get("SARASHINA_QUERY_MARK", "[[sarashina:query]]")
DOCUMENT_MARK = os.environ.get("SARASHINA_DOCUMENT_MARK", "[[sarashina:document]]")
# 印が無い入力の扱い。ふつうの OpenAI クライアントは印を付けないので、保存側として扱う
DEFAULT_TASK = os.environ.get("SARASHINA_DEFAULT_TASK", "RETRIEVAL_DOCUMENT")
RETRY_LIMIT_SECONDS = float(os.environ.get("SARASHINA_RETRY_LIMIT_SECONDS", "65"))
TIMEOUT_SECONDS = float(os.environ.get("SARASHINA_TIMEOUT_SECONDS", "30"))


def api_key():
    return os.environ.get("SARASHINA_API_KEY", "").strip()


class UpstreamError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def split_mark(text):
    """文頭の印を外して (本文, task_type) を返す。"""
    if text.startswith(QUERY_MARK):
        return text[len(QUERY_MARK):], "RETRIEVAL_QUERY"
    if text.startswith(DOCUMENT_MARK):
        return text[len(DOCUMENT_MARK):], "RETRIEVAL_DOCUMENT"
    return text, DEFAULT_TASK


def normalize(vector):
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]


def embed_one(text):
    """1件送ってベクトルと使ったトークン数を返す。429 は上限の秒数まで待って送り直す。"""
    body, task_type = split_mark(text)
    payload = json.dumps({"input": body, "task_type": task_type, "model": MODEL}).encode()
    deadline = time.monotonic() + RETRY_LIMIT_SECONDS
    while True:
        request = urllib.request.Request(
            f"{UPSTREAM}/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                result = json.load(response)
            return normalize(result["data"][0]["embedding"]), result.get("usage", {}).get("prompt_tokens", 0)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code != 429:
                raise UpstreamError(error.code, detail)
            try:
                wait = float(error.headers.get("Retry-After", "1"))
            except ValueError:
                wait = 1.0
            if time.monotonic() + wait > deadline:
                raise UpstreamError(429, detail)
            print(f"429: {wait:.0f} 秒待つ", file=sys.stderr, flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as error:
            raise UpstreamError(502, f"上流に繋がらない: {error}")


def encode_vector(vector, encoding_format):
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
    return vector


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "sarashina_proxy", "code": status}})

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "api_key_configured": bool(api_key())})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "sarashina"}]})
        else:
            self._error(404, "not found")

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self._error(404, "not found")
            return
        if not api_key():
            self._error(503, "SARASHINA_API_KEY が未設定（~/opt/sarashina-docker/.env に書いて up -d し直す）")
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except ValueError:
            self._error(400, "本文が JSON ではない")
            return
        texts = request.get("input")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            self._error(400, "input は文字列か文字列の配列で渡す（トークン列は未対応）")
            return
        encoding_format = request.get("encoding_format", "float")
        data = []
        total_tokens = 0
        try:
            for index, text in enumerate(texts):
                vector, tokens = embed_one(text)
                total_tokens += tokens
                data.append({"object": "embedding", "index": index, "embedding": encode_vector(vector, encoding_format)})
        except UpstreamError as error:
            self._error(error.status, error.message)
            return
        self._json(200, {
            "object": "list",
            "model": MODEL,
            "data": data,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        })

    def log_message(self, format, *args):
        print(f"{self.address_string()} {format % args}", file=sys.stderr, flush=True)


def main():
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler).serve_forever()


if __name__ == "__main__":
    main()
