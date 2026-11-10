import json, time, pathlib
import re as regex
from app.config import get_model, get_collection

def norm(s): return regex.sub(r"[\s\-]+", "", s.lower())
def is_hit(h, q):
    m = h["meta"]
    return q["source"] in m["source"] and any(norm(k) in norm(m["text"]) for k in q["keywords"])

qs = json.loads(pathlib.Path("eval/questions.json").read_text(encoding="utf-8"))
vecs = get_model().encode([q["question"] for q in qs]).tolist()
col = get_collection("docs_c150")
col.search(fields={"embedding": {"query": vecs[0], "limit": 5}})  # warm-up

out = ["| top_k | Hit@k | p50 ms |", "|---|---|---|"]
for k in [1, 3, 5, 10, 20]:
    found, lat = 0, []
    for q, v in zip(qs, vecs):
        t = time.perf_counter()
        hits = col.search(fields={"embedding": {"query": v, "limit": k}})["results"]["embedding"]
        lat.append((time.perf_counter() - t) * 1000)
        found += any(is_hit(h, q) for h in hits)
    lat.sort()
    out.append(f"| {k} | {found/len(qs):.2f} | {lat[len(lat)//2]:.0f} |")
text = "\n".join(out)
print(text)
pathlib.Path("eval/topk_results.md").write_text(text, encoding="utf-8")