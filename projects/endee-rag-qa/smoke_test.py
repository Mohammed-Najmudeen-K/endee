import os
from dotenv import load_dotenv
from endee import Endee
from sentence_transformers import SentenceTransformer

load_dotenv()
URL = os.environ["ENDEE_URL"]

admin = Endee(token=os.environ["ENDEE_ROOT_TOKEN"])
admin.set_base_url(URL)

try:
    db_token = admin.create_database("ragdb", db_type="enterprise")
    print("DB token (save this in .env as ENDEE_DB_TOKEN):", db_token)
except Exception as e:
    print("create_database failed (maybe it already exists):", e)
    db_token = os.environ["ENDEE_DB_TOKEN"]

client = Endee(token=db_token)
client.set_base_url(URL)

try:
    client.create_collection(
        name="smoke",
        fields=[{"name": "embedding", "type": "vector",
                 "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}}],
    )
except Exception as e:
    print("create_collection:", e)
col = client.get_collection("smoke")

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["Endee is a vector database", "Cats love sleeping", "RAG uses retrieval with LLMs"]
vecs = model.encode(texts).tolist()

col.upsert([
    {"id": f"d{i}", "meta": {"text": t}, "filter": {"category": "test"},
     "fields": {"embedding": v}}
    for i, (t, v) in enumerate(zip(texts, vecs))
])

q = model.encode("what is a vector store?").tolist()
res = col.search(fields={"embedding": {"query": q, "limit": 2}})
for item in res["results"]["embedding"]:
    print(item["id"], round(item["similarity"], 3), item.get("meta"))