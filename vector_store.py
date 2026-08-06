# from sklearn.metrics.pairwise import cosine_similarity

# def search(query_embedding, chunks, k):

#     scores = []

#     # Convert (384,) -> (1, 384)
#     query_embedding = query_embedding.reshape(1, -1)

#     for chunk in chunks:

#         # Convert (384,) -> (1, 384)
#         chunk_embedding = chunk["embedding"].reshape(1, -1)

#         # cosine_similarity returns [[score]]
#         score = cosine_similarity(query_embedding, chunk_embedding)[0][0]

#         scores.append((score, chunk))

#     # Sort only using the score
#     scores.sort(key=lambda x: x[0], reverse=True)

#     # Return only the top-k chunks
#     return [chunk for _, chunk in scores[:k]]