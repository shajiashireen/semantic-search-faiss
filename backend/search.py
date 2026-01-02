import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

index = faiss.read_index("data/faiss.index")
index.nprobe = 10

df = pd.read_csv("data/texts.csv") 


model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(query, top_k=5):
    query_vec = model.encode([query], normalize_embeddings=True)

    scores, ids = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        row = df.iloc[idx]
        results.append((row["text"], float(score), row["sentiment"]))

    return results
