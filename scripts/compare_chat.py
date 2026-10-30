#!/usr/bin/env python3
"""事実抜き出し（JSON スキーマ指定）を自作の文で比べる。個人情報を含まない固定の文だけを送る。
環境変数 DEEPSEEK_URL、DEEPSEEK_KEY があれば DeepSeek Flash も比べる。"""
import json, os, time, urllib.request, urllib.error
TEXT = """サーバーの移行作業を土曜の夜に行った。PostgreSQL は 15 から 17 に上げ、バックアップは事前に取得した。
移行中に拡張機能 pgvector の再インストールが必要だと分かり、30 分ほど遅れた。
今後は監視を Prometheus に寄せ、夜間のバッチは毎日 3 時に動かす方針にした。
なお、古い Grafana のダッシュボードは来月まで残す。"""
SCHEMA = {"name": "facts", "strict": True, "schema": {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "object",
  "properties": {"what": {"type": "string"}, "type": {"type": "string", "enum": ["world", "experience"]}}, "required": ["what", "type"], "additionalProperties": False}}},
  "required": ["facts"], "additionalProperties": False}}
MESSAGES = [{"role": "system", "content": "文章から、後で思い出す価値のある事実を日本語で1件ずつ抜き出す。具体的な数値や固有名詞を落とさない。"}, {"role": "user", "content": TEXT}]
TOOLS = [{"type": "function", "function": {"name": "search_memory", "description": "記憶を検索する", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
def call(name, url, key, model, **extra):
    body = dict(model=model, messages=MESSAGES, max_tokens=2000, **extra)
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "curl/8", "x-opencode-session": "sarashina-compare"})
    t = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"{name}: HTTP {e.code} {e.read()[:200]!r}"); return
    m = d["choices"][0]["message"]; dt = time.monotonic() - t
    return m, dt, d.get("usage", {})
targets = [("nano", "http://127.0.0.1:8082/v1/chat/completions", "x", "sarashina3-nano"), ("mini", "http://127.0.0.1:8082/v1/chat/completions", "x", "sarashina3-mini")]
if os.environ.get("DEEPSEEK_KEY"):
    targets.append(("deepseek-flash", os.environ["DEEPSEEK_URL"].rstrip("/") + "/chat/completions", os.environ["DEEPSEEK_KEY"], "deepseek-v4.1-flash"))
for name, url, key, model in targets:
    r = call(name, url, key, model, response_format={"type": "json_schema", "json_schema": SCHEMA}, temperature=0)
    if r:
        m, dt, u = r
        try:
            facts = json.loads(m["content"])["facts"]; ok = "JSON OK"
        except Exception as e:
            facts = []; ok = f"JSON NG {e}"
        print(f"== {name} {dt:.2f}s {ok} 件数={len(facts)} tokens={u.get('completion_tokens')}")
        for f in facts: print("  ", f.get("type"), f.get("what"))
    r = call(name, url, key, model, tools=TOOLS, tool_choice="required")
    if r:
        m, dt, u = r
        print(f"   tools {dt:.2f}s calls={[(c['function']['name'], c['function']['arguments'][:60]) for c in (m.get('tool_calls') or [])]}")
