"""LlamaIndex vector retrieval over local banking documents."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

documents = [
    Document(text="Alice Rivera wants a mortgage. Her credit score is 720."),
    Document(text="Ben Carter wants a personal loan. His credit score is 610."),
    Document(text="Bank policy says DTI above 45 percent requires manual review."),
]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)

retriever = index.as_retriever(similarity_top_k=2)

question = "Who wants a home loan?"

results = retriever.retrieve(question)

print(f"Question: {question}\n")

for i, result in enumerate(results, start=1):
    print(f"Result {i}")
    print(f"Score: {result.score}")
    print(f"Retrieved chunk: {result.text}")
    print()
