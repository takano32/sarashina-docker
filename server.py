#!/usr/bin/env python3
"""Sarashina の API を OpenAI 互換などの形で出す変換サーバー。

口は /v1/chat/completions、/v1/completions、/v1/embeddings、/v1/moderations、/v1/rerank（/rerank も同じ）、
/v1/translate、/v1/models。対応表は README。

/v1/chat/completions は次を直して中継する。
- max_tokens を max_completion_tokens に読み替えて、上限 8000 に丸める（max_tokens は上流で 400）
- logprobs、top_logprobs を落とす（上流で 400 の報告あり）
- 特定の関数を指す tool_choice を required に読み替える（オブジェクト形式は上流で 400）
- tools があって tool_choice が無いときは auto を明示する（省くと required で動く報告あり）
- 429 は Retry-After のぶん待って送り直す

/v1/embeddings で埋める違いは次のとおり（理由は README）。
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
CHAT_TIMEOUT_SECONDS = float(os.environ.get("SARASHINA_CHAT_TIMEOUT_SECONDS", "300"))
CHAT_MAX_COMPLETION_TOKENS = 8000
RERANK_MODEL = os.environ.get("SARASHINA_RERANK_MODEL", "sarashina3-rerank")
GUARD_MODEL = os.environ.get("SARASHINA_GUARD_MODEL", "sarashina3-guard")
RERANK_BATCH = 20
MODELS = (MODEL, RERANK_MODEL, GUARD_MODEL, "sarashina3-mini", "sarashina3-nano", "sarashina3.2-mini-thinking", "sarashina3-translate")


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


def adapt_chat(request):
    """OpenAI の形のチャットのリクエストを Sarashina が受ける形に直す。"""
    adapted = dict(request)
    limit = adapted.pop("max_tokens", None)
    limit = adapted.get("max_completion_tokens", limit)
    if limit is not None:
        adapted["max_completion_tokens"] = min(int(limit), CHAT_MAX_COMPLETION_TOKENS)
    adapted.pop("logprobs", None)
    adapted.pop("top_logprobs", None)
    if isinstance(adapted.get("tool_choice"), dict):
        adapted["tool_choice"] = "required"
    if adapted.get("tools") and "tool_choice" not in adapted:
        adapted["tool_choice"] = "auto"
    return adapted


def open_upstream(path, payload):
    """上流に送って応答を開く。429 は上限の秒数まで待って送り直す。呼び出し側が閉じる。"""
    deadline = time.monotonic() + RETRY_LIMIT_SECONDS
    while True:
        request = urllib.request.Request(
            f"{UPSTREAM}{path}",
            data=payload,
            headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(request, timeout=CHAT_TIMEOUT_SECONDS)
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


def post_json(path, body):
    with open_upstream(path, json.dumps(body).encode()) as response:
        return json.load(response)


def rerank(request):
    """Cohere／Jina の形で受けて返す。TEI の texts も受ける。20 件ずつに分けて送り、スコアの高い順に並べ直す。"""
    documents = request.get("documents", request.get("texts"))
    query = request.get("query")
    if not isinstance(query, str) or not isinstance(documents, list) or not documents:
        raise UpstreamError(400, "query と documents（または texts）を渡す")
    texts = [d["text"] if isinstance(d, dict) else d for d in documents]
    body = {"query": query, "model": RERANK_MODEL}
    for key in ("task_type", "query_instruction"):
        if key in request:
            body[key] = request[key]
    results = []
    tokens = 0
    for start in range(0, len(texts), RERANK_BATCH):
        answer = post_json("/v1/rerank", {**body, "documents": texts[start:start + RERANK_BATCH]})
        tokens += answer.get("usage", {}).get("prompt_tokens", 0)
        for item in answer["data"][0]["result"]:
            results.append({"index": start + item["document_id"], "relevance_score": item["score"]})
    results.sort(key=lambda r: -r["relevance_score"])
    top_n = request.get("top_n")
    if isinstance(top_n, int) and top_n > 0:
        results = results[:top_n]
    if request.get("return_documents"):
        for result in results:
            result["document"] = {"text": texts[result["index"]]}
    return {"object": "list", "model": RERANK_MODEL, "results": results, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


def moderations(request):
    """OpenAI の moderations の形で返す。sarashina3-guard の答えは 1 行目が 安全／有害、有害なら次の行がカテゴリ。"""
    inputs = request.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs or not all(isinstance(t, str) for t in inputs):
        raise UpstreamError(400, "input は文字列か文字列の配列で渡す")
    results = []
    for text in inputs:
        answer = post_json("/v1/chat/completions", {"model": GUARD_MODEL, "messages": [{"role": "user", "content": text}]})
        lines = answer["choices"][0]["message"]["content"].strip().splitlines()
        flagged = bool(lines) and lines[0].strip() == "有害"
        categories = {line.strip(): True for line in lines[1:] if line.strip()} if flagged else {}
        results.append({"flagged": flagged, "categories": categories, "category_scores": {k: 1.0 for k in categories}})
    return {"id": "modr-sarashina", "model": GUARD_MODEL, "results": results}


def completions(request):
    """古い /v1/completions の形をチャットに読み替える（prompt は 1 件のみ、ストリームなし）。"""
    prompt = request.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1:
        prompt = prompt[0]
    if not isinstance(prompt, str) or request.get("stream"):
        raise UpstreamError(400, "prompt は文字列 1 件で渡す（ストリームは /v1/chat/completions を使う）")
    body = {k: v for k, v in request.items() if k != "prompt"}
    body["messages"] = [{"role": "user", "content": prompt}]
    answer = post_json("/v1/chat/completions", adapt_chat(body))
    choices = [{"index": c["index"], "text": c["message"].get("content") or "", "finish_reason": c.get("finish_reason"), "logprobs": None} for c in answer["choices"]]
    return {"id": answer.get("id"), "object": "text_completion", "created": answer.get("created"), "model": answer.get("model"), "choices": choices, "usage": answer.get("usage")}


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
            self._json(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "sarashina"} for m in MODELS]})
        else:
            self._error(404, "not found")

    def _chat(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except ValueError:
            self._error(400, "本文が JSON ではない")
            return
        try:
            response = open_upstream("/v1/chat/completions", json.dumps(adapt_chat(request)).encode())
        except UpstreamError as error:
            self._error(error.status, error.message)
            return
        with response:
            self.send_response(200)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()

    def _simple(self, convert):
        """受けた JSON を convert に渡して、返り値を JSON で返す。"""
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except ValueError:
            self._error(400, "本文が JSON ではない")
            return
        try:
            self._json(200, convert(request))
        except UpstreamError as error:
            self._error(error.status, error.message)

    def do_POST(self):
        simple = {
            "/v1/rerank": rerank,
            "/rerank": rerank,
            "/v1/moderations": moderations,
            "/v1/completions": completions,
            "/v1/translate": lambda request: post_json("/v1/translate", request),
        }
        if self.path not in simple and self.path not in ("/v1/embeddings", "/v1/chat/completions"):
            self._error(404, "not found")
            return
        if not api_key():
            self._error(503, "SARASHINA_API_KEY が未設定（~/opt/sarashina-docker/.env に書いて up -d し直す）")
            return
        if self.path == "/v1/chat/completions":
            self._chat()
            return
        if self.path in simple:
            self._simple(simple[self.path])
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
