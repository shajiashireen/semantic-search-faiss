import faiss
import numpy as np
import time

embeddings = np.load("data/embeddings.npy").astype("float32")
dim = embeddings.shape[1]

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(
    quantizer,
    dim,
    100,
    faiss.METRIC_INNER_PRODUCT
)

print("Training FAISS index...")
start = time.time()
index.train(embeddings)
index.add(embeddings)
end = time.time()

faiss.write_index(index, "data/faiss.index")

print(f"FAISS index built in {end - start:.2f} seconds")
