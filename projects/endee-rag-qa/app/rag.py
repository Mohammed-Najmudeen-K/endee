from app.retriever import retrieve
from app.llm import generate

SYSTEM = ("Answer using ONLY the numbered context passages. "
          "Cite passages like [1], [2]. If the answer is not in the context, say you don't know.")

def answer(question, top_k=5, doc=None, collection="docs"):
    hits = retrieve(question, top_k, doc, collection)
    ctx = "\n\n".join(
        f"[{i}] ({h['meta']['source']}, p{h['meta']['page']})\n{h['meta']['text']}"
        for i, h in enumerate(hits, 1))
    try:
        text = generate(f"Context:\n{ctx}\n\nQuestion: {question}", SYSTEM)
    except Exception:
        text = "The LLM is temporarily unavailable. Showing the retrieved passages only."
    return {
        "answer": text,
        "sources": [{"n": i, "source": h["meta"]["source"], "page": h["meta"]["page"],
                     "score": round(h["similarity"], 3), "text": h["meta"]["text"]}
                    for i, h in enumerate(hits, 1)],
    }