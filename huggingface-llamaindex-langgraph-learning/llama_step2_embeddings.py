from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

texts = [
    "Alice wants a mortgage.",
    "Alice is applying for a home loan.",
    "The weather is rainy today.",
]

embeddings = [
    embed_model.get_text_embedding(text)
    for text in texts
]

print("Similarity scores:\n")

print(
    "mortgage vs home loan:",
    cosine_similarity(embeddings[0], embeddings[1])
)

print(
    "mortgage vs weather:",
    cosine_similarity(embeddings[0], embeddings[2])
)