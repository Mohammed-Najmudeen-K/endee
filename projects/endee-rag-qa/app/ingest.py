import argparse, pathlib
from pypdf import PdfReader
from app.config import get_model, get_collection
from app.chunking import chunk_text

def load_pages(path):
    if path.suffix.lower() == ".pdf":
        for i, p in enumerate(PdfReader(str(path)).pages, 1):
            yield i, p.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--collection", default="docs")
    a = ap.parse_args()

    col = get_collection(a.collection, create=True)
    model = get_model()
    items = []
    for path in sorted(pathlib.Path(a.folder).glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        for page, text in load_pages(path):
            for n, chunk in enumerate(chunk_text(text, a.chunk, a.overlap)):
                items.append((f"{path.stem}-p{page}-c{n}", chunk, path.name, page))
    print(f"{len(items)} chunks")

    for s in range(0, len(items), 100):
        batch = items[s:s + 100]
        vecs = model.encode([b[1] for b in batch], batch_size=16).tolist()
        col.upsert([
            {"id": id_, "meta": {"text": txt, "source": src, "page": pg},
             "filter": {"doc": src},
             "fields": {"embedding": v}}
            for (id_, txt, src, pg), v in zip(batch, vecs)
        ])
    print("done")

if __name__ == "__main__":
    main()