"""
Step 3: Retriever demo.

This script shows the RAG pipeline up to retrieval:

documents/chunks -> embeddings -> vector index -> retriever -> top matching chunks

There is no LLM answer yet. The goal is only to see which chunks get returned.
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ---------------------------------------------------------------------------
# 1. DOCUMENTS / CHUNKS
# ---------------------------------------------------------------------------
# In a real RAG app, you might start with a long PDF or text file, then split it
# into chunks using a chunker like SentenceSplitter.
#
# Here, we skip the chunking step and manually create 3 short "chunks" directly.
# Each Document below acts like one searchable chunk.
documents = [
    Document(text="Alice Rivera wants a mortgage. Her credit score is 720."),
    Document(text="Ben Carter wants a personal loan. His credit score is 610."),
    Document(text="Bank policy says DTI above 45 percent requires manual review."),
]


# ---------------------------------------------------------------------------
# 2. EMBEDDER
# ---------------------------------------------------------------------------
# The embedder turns text into vectors.
#
# It will embed:
# - each chunk/document above
# - the user question later
#
# Similar meanings should produce vectors that point in similar directions.
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# ---------------------------------------------------------------------------
# 3. VECTOR INDEX
# ---------------------------------------------------------------------------
# The index stores the chunks and their embeddings.
#
# When this line runs, LlamaIndex:
# 1. takes each Document
# 2. embeds its text with embed_model
# 3. stores the text + vector in an in-memory vector index
#
# This is the searchable database for this tiny example.
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)


# ---------------------------------------------------------------------------
# 4. RETRIEVER
# ---------------------------------------------------------------------------
# The retriever searches the vector index.
#
# similarity_top_k=2 means:
# "Return the top 2 chunks whose embeddings are closest to the question."
#
# The retriever does not generate an answer. It only returns matching chunks.
retriever = index.as_retriever(
    similarity_top_k=2
)


# ---------------------------------------------------------------------------
# 5. USER QUESTION
# ---------------------------------------------------------------------------
# This is the query we want to search with.
#
# Notice the question says "home loan", but the chunk says "mortgage".
# Embeddings should understand that these are semantically similar.
question = "Who wants a home loan?"


# ---------------------------------------------------------------------------
# 6. RETRIEVAL
# ---------------------------------------------------------------------------
# When retrieve() runs, LlamaIndex:
# 1. embeds the question
# 2. compares the question vector to all chunk vectors
# 3. returns the top_k most similar chunks
results = retriever.retrieve(question)


# ---------------------------------------------------------------------------
# 7. PRINT RESULTS
# ---------------------------------------------------------------------------
# These are the chunks that would later be passed into an LLM.
#
# For now, we print them so you can inspect whether retrieval worked.
print(f"Question: {question}\n")

for i, result in enumerate(results, start=1):
    print(f"Result {i}")
    print(f"Score: {result.score}")
    print(f"Retrieved chunk: {result.text}")
    print()
