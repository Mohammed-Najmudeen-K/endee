import json, pathlib
import re as regex
from pypdf import PdfReader
from app.config import get_model, get_collection
from app.chunking import chunk_text

def norm(s): return regex.sub(r"[\s\-]+", "", s.lower())

qs = json.loads(pathlib.Path("eval/questions.json").read_text(encoding="utf-8"))
targets = [q for q in qs if q["question"].startswith(
    ("How are the layers", "What attention function"))]
col = get_collection("docs_c150")

for q in targets:
    kw = q["keywords"][0]
    print("\nQ:", q["question"], "| keyword:", kw)

    # 1) rank of the first chunk containing the keyword, searching top 50
    v = get_model().encode(q["question"]).tolist()
    hits = col.search(fields={"embedding": {"query": v, "limit": 50}})["results"]["embedding"]
    rank = next((i for i, h in enumerate(hits, 1)
                 if q["source"] in h["meta"]["source"] and norm(kw) in norm(h["meta"]["text"])), None)
    print("  rank of first intact chunk in top 50:", rank)

    # 2) do any intact chunks exist at chunk size 150?
    pdf = next(pathlib.Path("data").glob(f"{q['source']}*.pdf"))
    intact = []
    for page, p in enumerate(PdfReader(str(pdf)).pages, 1):
        for n, c in enumerate(chunk_text(p.extract_text() or "", 150, 30)):
            if norm(kw) in norm(c):
                intact.append(f"p{page}-c{n}")
    print("  chunks containing the keyword intact:", intact or "NONE (phrase is split)")