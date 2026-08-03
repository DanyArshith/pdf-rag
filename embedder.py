from model import embedding_model 

def generate_embeddings(chunks):
    for chunk in chunks:
        chunk["embedding"] = embedding_model.encode(chunk["text"])
    return chunks