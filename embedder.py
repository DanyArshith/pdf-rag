from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")


embedding = model.encode("hello this is a sample text")

def generate_embeddings(chunks):
    for chunk in chunks:
        chunk["embedding"] = model.encode(chunk["text"])

    return chunks