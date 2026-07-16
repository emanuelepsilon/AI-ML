"""
Step 4: Query engine demo.

This script shows the first full RAG loop:

documents/chunks -> embeddings -> vector index -> retriever -> LLM answer

The difference from step 3:
- Step 3 only printed retrieved chunks.
- Step 4 gives those chunks to an LLM and gets a final natural-language answer.
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. DOCUMENTS / CHUNKS
# ---------------------------------------------------------------------------
# These are our tiny knowledge base.
#
# In a real app, these would usually come from:
# - text files
# - PDFs
# - databases
# - support tickets
# - company policies
#
# Here each Document is already small, so each one acts like one chunk.
documents = [
    Document(text="Alice Rivera wants a mortgage. Her credit score is 720."),
    Document(text="Ben Carter wants a personal loan. His credit score is 610."),
    Document(text="Bank policy says DTI above 45 percent requires manual review."),
]


# ---------------------------------------------------------------------------
# 2. EMBEDDER
# ---------------------------------------------------------------------------
# The embedder turns each document/chunk into a semantic vector.
#
# Later, the user's question will also be embedded and compared to these
# chunk vectors.
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# ---------------------------------------------------------------------------
# 3. VECTOR INDEX
# ---------------------------------------------------------------------------
# The index stores:
# - chunk text
# - chunk embeddings/vectors
# - metadata
#
# This is the searchable vector space of the knowledge base.
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)


# ---------------------------------------------------------------------------
# 4. LOCAL LLM
# ---------------------------------------------------------------------------
# This is the answer-writing model.
#
# It talks to your local Ollama server, so Ollama must be running.
#
# Terminal 1:
#   ollama serve
#
# If Ollama is already running in the background, you do not need to start it.
llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 5. QUERY ENGINE
# ---------------------------------------------------------------------------
# The query engine combines retrieval + LLM answering.
#
# Under the hood, it:
# 1. embeds the question
# 2. retrieves the top matching chunks from the index
# 3. places those chunks into the LLM prompt
# 4. asks the LLM to write the final answer
#
# similarity_top_k=2 means the LLM sees the top 2 retrieved chunks.
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=2,
)


# ---------------------------------------------------------------------------
# 6. ASK A QUESTION
# ---------------------------------------------------------------------------
# This question uses "home loan", while the document says "mortgage".
# The retriever should still find Alice's mortgage chunk because embeddings
# capture semantic similarity.
question = "Who wants a home loan, and what is their credit score?"

response = query_engine.query(question)


# ---------------------------------------------------------------------------
# 7. PRINT FINAL ANSWER
# ---------------------------------------------------------------------------
# This is now an LLM-written answer.
#
# The answer should be grounded in the chunks that the query engine retrieved.
print(f"Question: {question}\n")
print("Answer:")
print(response)


# ---------------------------------------------------------------------------
# 8. PRINT SOURCES / RETRIEVED CHUNKS
# ---------------------------------------------------------------------------
# This lets you inspect what the LLM actually received as context.
#
# Each source node is one retrieved chunk.
# The score shows how similar that chunk was to the question.
#
# This is important for debugging RAG:
# - If the answer is wrong, inspect these chunks first.
# - If the right chunk is missing, retrieval failed.
# - If the right chunk is present but the answer is wrong, synthesis failed.
print("\nSources used:")

for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Score: {source_node.score}")
    print(f"Chunk text: {source_node.node.text}")
