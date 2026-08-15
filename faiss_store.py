import faiss
import numpy as np
from pathlib import Path
import pickle


STORAGE_DIR = Path("data/storage")
INDEX_PATH = STORAGE_DIR / "index.faiss"
CHUNKS_PATH = STORAGE_DIR / "chunks.pkl"


def build_index(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(chunk["embedding"])

    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def save_index(index, chunks):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)


def load_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search(index, query_embedding, chunks, k):
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_embedding, k)

    return [chunks[index] for index in indices[0]]