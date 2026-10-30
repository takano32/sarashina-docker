#!/usr/bin/env python3
"""手元の埋め込みサーバーと sarashina-embed を、自作の日本語文の小さな検索問題で比べる。
python3 scripts/compare_retrieval.py [本数の上限なし]。個人情報を含まない固定の文だけを送る。"""
import json, math, statistics, time, urllib.request

DOCS = {
 "wal": "PostgreSQL の WAL は、変更をデータファイルへ書く前にログへ書き出す仕組みで、障害後の復旧に使う。",
 "vacuum": "VACUUM は不要になった行の領域を再利用できるようにする処理で、肥大化を防ぐ。",
 "oom": "メモリが足りなくなると Linux の OOM killer がプロセスを選んで強制終了する。",
 "swap": "スワップは物理メモリから追い出したページをディスクに置く領域で、速度は大きく落ちる。",
 "compose": "docker compose の depends_on に service_healthy を指定すると、依存先の準備ができてから起動する。",
 "dns": "コンテナ同士は同じネットワークに入っていればサービス名で名前解決できる。",
 "sourdough": "サワードウは小麦粉と水を発酵させた種を使うパンで、独特の酸味がある。",
 "miso": "味噌は大豆と麹と塩を発酵させて作り、地域によって色や味が違う。",
 "ramen": "豚骨ラーメンは豚の骨を長時間煮込んだ白濁したスープが特徴だ。",
 "okinawa": "沖縄は梅雨入りが本州より早く、例年5月ごろに雨の季節が始まる。",
 "hokkaido": "北海道の冬は気温が氷点下になる日が多く、道路の凍結に注意が必要だ。",
 "typhoon": "台風は夏から秋にかけて日本に近づき、強風と大雨をもたらす。",
 "bike": "自転車のチェーンは定期的に注油しないと錆びて変速が悪くなる。",
 "tire": "タイヤの空気圧が低いと燃費が悪化し、パンクもしやすくなる。",
 "train": "新幹線は東京と新大阪の間を最速で2時間半ほどで結ぶ。",
 "edo": "江戸時代の参勤交代では、大名が一年おきに江戸と領地を行き来した。",
 "meiji": "明治維新で武士の身分は廃止され、近代的な中央集権国家が作られた。",
 "photosyn": "光合成では植物が光のエネルギーで二酸化炭素と水から糖を作る。",
 "vaccine": "ワクチンは弱めた病原体などで免疫に記憶させ、感染症を防ぐ。",
 "blackhole": "ブラックホールは重力が強すぎて光さえ脱出できない天体である。",
}
QUERIES = [
 ("クラッシュしたときデータベースはどうやって元に戻すの", "wal"),
 ("テーブルが膨らんでしまうのを防ぐ掃除の処理", "vacuum"),
 ("サーバーでプロセスが急に殺された、メモリ不足かも", "oom"),
 ("RAM が足りないときディスクに逃がす機能", "swap"),
 ("DB が起動するまでアプリの起動を待たせたい", "compose"),
 ("コンテナから別のコンテナにホスト名で繋ぐには", "dns"),
 ("酸っぱいパンの作り方に使う種", "sourdough"),
 ("麹を使う日本の調味料", "miso"),
 ("スープが白いラーメン", "ramen"),
 ("雨季が早く来る南の島", "okinawa"),
 ("路面が滑りやすくなる寒い地方", "hokkaido"),
 ("秋に来る暴風雨", "typhoon"),
 ("ギアの切り替えが渋いのはなぜ", "bike"),
 ("ガソリン代が増えた、空気が抜けてるのかも", "tire"),
 ("東京から大阪まで鉄道で何時間", "train"),
 ("殿様が定期的に首都へ行く制度", "edo"),
 ("侍がいなくなったのはいつ", "meiji"),
 ("葉っぱが太陽の光で栄養を作るしくみ", "photosyn"),
 ("予防接種はどうして効くのか", "vaccine"),
 ("光も逃げられない星", "blackhole"),
]
def embed(url, model, text):
    req = urllib.request.Request(url, json.dumps({"model": model, "input": [text]}).encode(), {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        v = json.load(r)["data"][0]["embedding"]
    return v, time.monotonic() - t
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))
def run(name, url, model, qmark="", dmark=""):
    lat = []
    docs = {}
    for k, d in DOCS.items():
        docs[k], s = embed(url, model, dmark + d); lat.append(s)
    hit1 = hit3 = 0; rr = 0.0; misses = []; margins = []
    for q, ans in QUERIES:
        v, s = embed(url, model, qmark + q); lat.append(s)
        ranked = sorted(docs, key=lambda k: -cos(v, docs[k]))
        r = ranked.index(ans) + 1
        hit1 += r == 1; hit3 += r <= 3; rr += 1 / r
        if r > 1: misses.append(f"{q} -> {ranked[0]}({r}位)")
    n = len(QUERIES)
    print(f"{name}: dim={len(next(iter(docs.values())))} top1={hit1}/{n} top3={hit3}/{n} MRR={rr/n:.3f} 遅延中央値={statistics.median(lat)*1000:.0f}ms 最大={max(lat)*1000:.0f}ms")
    for m in misses: print("  miss:", m)
run("bge-m3", "http://127.0.0.1:8081/v1/embeddings", "bge-m3")
run("sarashina", "http://127.0.0.1:8082/v1/embeddings", "x", "[[sarashina:query]]", "[[sarashina:document]]")
