# Semantic Search Engine with FAISS

This project implements a Semantic Search Engine that retrieves text results based on meaning and context rather than exact keyword matching.
It uses Sentence Transformers for embedding generation and FAISS for fast similarity search, with a simple and interactive Streamlit UI.



## 🚀 Features

- Semantic similarity search using dense embeddings
- Fast Approximate Nearest Neighbor (ANN) search with FAISS
- Clean Streamlit-based web interface
- Displays top-k relevant results with similarity scores
- Efficient indexing and low query latency


#  Project Structure

```
semantic-search-faiss/
│
├── backend/
│   ├── embed.py            # Generate sentence embeddings
│   ├── build_index.py      # Build FAISS index
│   ├── search.py           # Query FAISS index
│   └── __init__.py
│
├── data/
│   ├── tweets.csv          # Raw dataset
│   ├── texts.csv           # Cleaned text data
│   ├── embeddings.npy      # Generated embeddings (ignored in git)
│   └── faiss.index         # FAISS index file (ignored in git)
│
├── ui/
│   └── app.py              # Streamlit UI
│
├── requirements.txt
├── .gitignore
└── README.md



#  Technologies Used

- Python 3.x
- sentence-transformers
- FAISS
- Pandas & NumPy
- Streamlit


#  Setup Instructions

# 1️ Clone the Repository

```bash
git clone https://github.com/shajiashireen/semantic-search-faiss.git
cd semantic-search-faiss
```



# 2️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  How to Run the Project

# Step 1: Generate Sentence Embeddings

```bash
python backend/embed.py
```

---

# Step 2: Build FAISS Index

```bash
python backend/build_index.py
```

---

# Step 3: Launch the Web Interface

```bash
streamlit run ui/app.py
```

---

#  Example Queries

- happy
- feeling tired today
- missing my friends
- need motivation

---

#  Performance Metrics

| Metric | Value |
|------|------|
| Dataset Size | 5,000 text snippets |
| Embedding Model | all-MiniLM-L6-v2 |
| Index Type | FAISS IVF |
| Indexing Time | ~1–2 seconds |
| Average Query Latency | ~3–6 ms |



#  How It Works

1. Text data is converted into numerical embeddings using a pretrained transformer model.
2. These embeddings are indexed using FAISS for fast similarity search.
3. User queries are embedded and compared against the index.
4. The most semantically similar results are returned with a similarity score.


#  Notes

- Semantic search focuses on context and meaning, not exact keywords.
- Similarity scores represent semantic closeness.

