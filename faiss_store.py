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

def search(index, query_embedding, chunks, k):
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_embedding, k)

    return [chunks[index] for index in indices[0]]