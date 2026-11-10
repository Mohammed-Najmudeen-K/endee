import sys, time
from app.config import get_model, get_collection

def retrieve(query, top_k=5, doc=None, collection="docs", ef=None):
    col = get_collection(collection)
    spec = {"query": get_model().encode(query).tolist(), "limit": top_k}
    if ef:
        spec["ef_search"] = ef
    kwargs = {"filter": [{"doc": {"$eq": doc}}]} if doc else {}
    res = col.search(fields={"embedding": spec}, **kwargs)
    return res["results"]["embedding"]

if __name__ == "__main__":
    t = time.perf_counter()
    for h in retrieve(" ".join(sys.argv[1:])):
        print(round(h["similarity"], 3), h["meta"]["source"], "p", h["meta"]["page"])
        print("  ", h["meta"]["text"][:120].replace("\n", " "))
    print(f"{(time.perf_counter()-t)*1000:.0f} ms")