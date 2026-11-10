import os
from dotenv import load_dotenv
from endee import Endee
from sentence_transformers import SentenceTransformer

load_dotenv()
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
    return _model

def get_collection(name="docs", create=False):
    client = Endee(token=os.environ["ENDEE_DB_TOKEN"])
    client.set_base_url(os.environ["ENDEE_URL"])
    if create:
        try:
            client.create_collection(
                name=name,
                fields=[{"name": "embedding", "type": "vector",
                         "params": {"dimension": 384, "space_type": "cosine",
                                    "precision": "int8"}}],
            )
        except Exception:
            pass  # already exists
    return client.get_collection(name)