import re, pathlib
from pypdf import PdfReader

def norm(s): return re.sub(r"[\s\-]+", "", s.lower())

checks = {"1603.09320": "exponentially decaying", "1706.03762": "Scaled Dot-Product"}
for stem, kw in checks.items():
    pdf = next(pathlib.Path("data").glob(f"{stem}*.pdf"))
    for i, p in enumerate(PdfReader(str(pdf)).pages, 1):
        t = p.extract_text() or ""
        raw, nrm = kw.lower() in t.lower(), norm(kw) in norm(t)
        if raw or nrm:
            print(stem, "page", i, "| exact match:", raw, "| ignoring spaces/hyphens:", nrm)