import faiss
import numpy as np

def build_index(chunks):

    embeddings = []
    for chunk in chunks:
        embeddings.append(chunk["embedding"])

    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index