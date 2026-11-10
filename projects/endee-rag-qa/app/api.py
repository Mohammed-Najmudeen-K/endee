import pathlib
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import answer

app = FastAPI(title="Endee RAG Q&A")

class Ask(BaseModel):
    question: str
    top_k: int = 5
    doc: Optional[str] = None

@app.post("/ask")
def ask(body: Ask):
    try:
        return answer(body.question, body.top_k, body.doc)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable, try again: {e}")

@app.get("/documents")
def documents():
    return sorted(p.name for p in pathlib.Path("data").glob("*.pdf"))