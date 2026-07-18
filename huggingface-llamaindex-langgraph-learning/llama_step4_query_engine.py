"""LlamaIndex query engine with local retrieval and synthesis."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

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

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=2,
)

question = "Who wants a home loan, and what is their credit score?"

response = query_engine.query(question)

print(f"Question: {question}\n")
print("Answer:")
print(response)

print("\nSources used:")

for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Score: {source_node.score}")
    print(f"Chunk text: {source_node.node.text}")
