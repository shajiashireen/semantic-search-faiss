import streamlit as st
import sys, os
if "results" not in st.session_state:
    st.session_state.results = []


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.search import semantic_search

st.set_page_config(page_title="Semantic Search Engine", layout="centered")

st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1f2933, #111827);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 14px;
    color: white;
    animation: fadeIn 0.5s ease-in;
}
.semantic {
    border-left: 6px solid #22c55e;
}
.emotion {
    border-left: 6px solid #60a5fa;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(8px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔍 Semantic Search Engine")
st.caption("FAISS + Sentence Transformers")

query = st.text_input("Enter your search query", placeholder="e.g. feeling happy today")

if query.strip():
    with st.spinner("Searching semantically..."):
        st.session_state.results = semantic_search(query)

    st.markdown("### 🔎 Top Semantic Matches")



    for i, (text, score, sentiment) in enumerate(st.session_state.results, 1):
        confidence = int(score * 100)

        st.markdown(f"""
        <div class="card">
            <b>Result {i}</b><br>
            {text}<br>
            <small>Match Strength: {confidence}%</small>
        </div>
        """, unsafe_allow_html=True)


    

st.markdown("---")
st.caption("Built using FAISS • Sentence-Transformers • Streamlit")
