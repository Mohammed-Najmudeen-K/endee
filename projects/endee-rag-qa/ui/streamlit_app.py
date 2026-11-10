import requests, streamlit as st

API = "http://localhost:8000"
st.title("Endee RAG Q&A")

docs = ["All documents"] + requests.get(f"{API}/documents").json()
doc = st.sidebar.selectbox("Search in", docs)
top_k = st.sidebar.slider("Top-k", 1, 10, 5)

q = st.text_input("Ask a question")
if q:
    with st.spinner("Thinking..."):
        resp = requests.post(f"{API}/ask", json={
            "question": q, "top_k": top_k,
            "doc": None if doc == "All documents" else doc})
    if resp.status_code != 200:
        st.error(f"{resp.status_code}: {resp.text[:500]}")
        st.stop()
    r = resp.json()
    st.markdown(r["answer"])
    st.subheader("Sources")
    for s in r["sources"]:
        with st.expander(f"[{s['n']}] {s['source']} p{s['page']} (score {s['score']})"):
            st.write(s["text"])