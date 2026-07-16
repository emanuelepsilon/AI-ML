"""
Step 5: Evaluation demo.

This script adds evaluation to the RAG pipeline.

Pipeline so far:

documents/chunks -> embeddings -> vector index -> query engine -> LLM answer

New part:

LLM answer + retrieved context -> evaluator -> pass/fail + feedback

The evaluator checks whether the answer is supported by the retrieved chunks.
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. DOCUMENTS / CHUNKS
# ---------------------------------------------------------------------------
# Same tiny knowledge base as before.
# Each Document acts like one chunk.
documents = [
    Document(text="Alice Rivera wants a mortgage. Her credit score is 720."),
    Document(text="Ben Carter wants a personal loan. His credit score is 610."),
    Document(text="Bank policy says DTI above 45 percent requires manual review."),
]


# ---------------------------------------------------------------------------
# 2. EMBEDDER
# ---------------------------------------------------------------------------
# Turns chunks and questions into semantic vectors.
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# ---------------------------------------------------------------------------
# 3. VECTOR INDEX
# ---------------------------------------------------------------------------
# Stores chunk text + chunk embeddings so retrieval can happen.
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)


# ---------------------------------------------------------------------------
# 4. LOCAL LLM
# ---------------------------------------------------------------------------
# This LLM is used twice:
# 1. to answer the question
# 2. to evaluate whether the answer is faithful to the retrieved context
#
# Make sure Ollama is running:
#   ollama serve
llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 5. QUERY ENGINE
# ---------------------------------------------------------------------------
# Retrieves the top chunks and asks the LLM to answer from them.
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=2,
)


# ---------------------------------------------------------------------------
# 6. ASK QUESTION
# ---------------------------------------------------------------------------
question = "Who wants a home loan, and what is their credit score?"
response = query_engine.query(question)


# ---------------------------------------------------------------------------
# 7. EVALUATOR
# ---------------------------------------------------------------------------
# FaithfulnessEvaluator checks:
#
# "Is the answer supported by the retrieved context?"
#
# It does not check whether the answer is useful, complete, or stylish.
# It checks whether the answer is grounded in the retrieved chunks.
evaluator = FaithfulnessEvaluator(llm=llm)
eval_result = evaluator.evaluate_response(
    query=question,
    response=response,
)


# ---------------------------------------------------------------------------
# 8. PRINT ANSWER, SOURCES, AND EVALUATION
# ---------------------------------------------------------------------------
print(f"Question: {question}\n")

print("Answer:")
print(response)

print("\nSources used:")
for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Score: {source_node.score}")
    print(f"Chunk text: {source_node.node.text}")

print("\nEvaluation:")
print(f"Passing: {eval_result.passing}")
print(f"Feedback: {eval_result.feedback}")
