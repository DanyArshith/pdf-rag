from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, results, top_k = 5):
    pairs = []

    for result in results:
        chunk = result['chunk']
        pairs.append((query, chunk["text"]))

    scores = reranker_model.predict(pairs)
    reranked = []

    for result, score in zip(results, scores):
        reranked.append({
            "chunk": result["chunk"],
            "distance": result["distance"],
            "rerank_score": float(score)
        })

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]