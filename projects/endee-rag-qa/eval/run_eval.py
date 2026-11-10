import json, time, pathlib
import re as regex
from app.config import get_model, get_collection

SIZES = [100, 150, 200]
K = 5
qs = json.loads(pathlib.Path("eval/questions.json").read_text(encoding="utf-8"))
vecs = get_model().encode([q["question"] for q in qs]).tolist()  # embedding not timed

def norm(s): return regex.sub(r"[\s\-]+", "", s.lower())

def is_hit(h, q):
    m = h["meta"]
    return q["source"] in m["source"] and any(
        norm(k) in norm(m["text"]) for k in q["keywords"])
rows = []
for size in SIZES:
    col = get_collection(f"docs_c{size}")
    col.search(fields={"embedding": {"query": vecs[0], "limit": K}})  # warm-up
    ranks, lat = [], []
    for q, v in zip(qs, vecs):
        t = time.perf_counter()
        res = col.search(fields={"embedding": {"query": v, "limit": K}})
        lat.append((time.perf_counter() - t) * 1000)
        hits = res["results"]["embedding"]
        ranks.append(next((i for i, h in enumerate(hits, 1) if is_hit(h, q)), None))
    n = len(qs)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    lat.sort()
    rows.append((size, hit(1), hit(3), hit(5), mrr, lat[n // 2], lat[min(n - 1, int(n * 0.95))]))

out = ["| Chunk (words) | Hit@1 | Hit@3 | Hit@5 | MRR | p50 ms | p95 ms |",
       "|---|---|---|---|---|---|---|"]
for s, a, b, c, m, p50, p95 in rows:
    out.append(f"| {s} | {a:.2f} | {b:.2f} | {c:.2f} | {m:.2f} | {p50:.0f} | {p95:.0f} |")
text = "\n".join(out)
print(text)
pathlib.Path("eval/results.md").write_text(text, encoding="utf-8")

col = get_collection("docs_c150")
for q, v in zip(qs, vecs):
    hits = col.search(fields={"embedding": {"query": v, "limit": K}})["results"]["embedding"]
    if not any(is_hit(h, q) for h in hits):
        print("MISS:", q["question"], "->", [h["meta"]["source"] for h in hits[:3]])