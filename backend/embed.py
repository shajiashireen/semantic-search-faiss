import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv(
    "data/tweets.csv",
    encoding="latin-1",
    header=None
)
df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']

df = df[['text', 'sentiment']].dropna().iloc[:5000]

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save("data/embeddings.npy", embeddings)
df.to_csv("data/texts.csv", index=False)

print("Embeddings and texts saved successfully!")
