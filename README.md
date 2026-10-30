# sarashina-docker

[Sarashina](https://docs.sarashina.ai/) の Embeddings API（`sarashina3-embedding`、1536次元）を、OpenAI 互換の `/v1/embeddings` で出すローカルの変換サーバー。
モデルは手元に置かず、計算は Sarashina 側で行う。標準ライブラリだけの `server.py` 1本。

| 項目 | 値 |
|---|---|
| compose のプロジェクト | `sarashina` |
| サービス／コンテナ | `embed`／`sarashina-embed` |
| 使う側から見る口 | external network `sarashina` 上の `http://sarashina-embed:8080/v1` |
| ホストからの口 | `127.0.0.1:8082`（`.env` の `SARASHINA_EMBED_PORT`。外には開けない） |
| 鍵 | `.env` の `SARASHINA_API_KEY`（git には入らない） |

## 使い方

```sh
cd ~/opt/sarashina-docker
cp .env.example .env && chmod 600 .env   # SARASHINA_API_KEY= の行に鍵を書く
docker compose up -d
curl -s http://127.0.0.1:8082/health     # api_key_configured が true なら鍵が入っている
```

鍵は <https://chat.sarashina.ai/dashboard/api-keys> で発行する。未設定のあいだ `/v1/embeddings` は 503 を返す。

## 埋めている違い

| Sarashina | このサーバー |
|---|---|
| `input` は1リクエスト1件 | 配列を1件ずつ順に送ってまとめて返す |
| `task_type` が必須 | 文頭の印から決めて、印は取り除く（下） |
| ベクトルは正規化されていない | 長さ1に正規化して返す |
| 60 回/分、超えると 429 | `Retry-After` のぶん待って送り直す（合計 65 秒まで。超えたら 429 を返す） |
| `encoding_format` は float のみ | `base64` にも応える（OpenAI の Python SDK の既定） |

配列の件数ぶんだけ 60 回/分の枠を使う。使う側のバッチは小さくしておく。

## 保存と検索の見分け

OpenAI の形には `task_type` を載せる欄が無いので、文頭の印で渡す。

| 文頭 | task_type |
|---|---|
| `[[sarashina:query]]` | `RETRIEVAL_QUERY` |
| `[[sarashina:document]]` | `RETRIEVAL_DOCUMENT` |
| 印なし | `SARASHINA_DEFAULT_TASK`（既定 `RETRIEVAL_DOCUMENT`） |

Hindsight なら設定だけで印を付けられる（まだ実物では試していない）。

```sh
HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX='[[sarashina:query]]'
HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX='[[sarashina:document]]'
```

## テスト

```sh
python3 scripts/test_server.py   # 偽の上流で確かめる。鍵も通信も要らない
```

**実物の API にはまだ当てていない。**

## 規約の注意

[利用規約](https://sarashina.ai/terms)（2026-10-30 に確認）の要点。

- 無償プランでは入出力がモデルの学習に使われ、断れるのは有償プランだけ（第11条）。
- 特定の個人を識別できる情報の入力は禁止（第10条）。
- 違反と判断されると、通知なしで API キーやアカウントを止められることがある（第6条）。

止められると埋め込みが止まるので、手元の埋め込みサーバーは消さずに戻れるようにしておく。
次元が違うモデルへ移るときは、保存済みのベクトルの埋め込み直しが要る。
