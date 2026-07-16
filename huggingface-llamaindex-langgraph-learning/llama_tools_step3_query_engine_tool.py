"""
LlamaIndex Tools - Step 3: QueryEngineTool.

Goal:
Wrap a RAG query engine as a tool, then give that tool to an agent.

Important idea:
FunctionTool wraps a normal Python function.
QueryEngineTool wraps a whole RAG query engine.

So the agent can call a knowledge base as a tool.
"""

import asyncio

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. DOCUMENTS / CHUNKS
# ---------------------------------------------------------------------------
# This is a tiny bank knowledge base.
#
# Each Document acts like a chunk that can be retrieved by semantic search.
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
# Stores the chunk text and embeddings in a searchable index.
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)


# ---------------------------------------------------------------------------
# 4. LOCAL LLM
# ---------------------------------------------------------------------------
# We use one local Ollama model for both:
# - the query engine's RAG answer
# - the agent's reasoning/tool choice
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
# The query engine does:
#
# question -> retrieve chunks -> send question + chunks to LLM -> answer
#
# This is the same RAG object from the previous lesson.
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=2,
)


# ---------------------------------------------------------------------------
# 6. QUERY ENGINE TOOL
# ---------------------------------------------------------------------------
# QueryEngineTool wraps the RAG query engine as a tool.
#
# The agent will see this tool by name and description.
# If the user asks about customers, loans, or policy, the agent should call it.
bank_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="bank_knowledge_base",
    description=(
        "Use this tool to answer questions about bank customers, loan products, "
        "credit scores, and bank risk policies."
    ),
)


# ---------------------------------------------------------------------------
# 7. AGENT WITH RAG TOOL
# ---------------------------------------------------------------------------
# The agent does not directly know the bank facts.
#
# It can answer bank questions by calling bank_knowledge_base.
agent = ReActAgent(
    tools=[bank_tool],
    llm=llm,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 8. ASK THE AGENT
# ---------------------------------------------------------------------------
# The agent should decide:
# "This is a bank knowledge question, so I should call bank_knowledge_base."
async def main():
    question = "Who wants a home loan, and what is their credit score?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
