def retrieve(query_embedding, index, chunks: list[str], top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]