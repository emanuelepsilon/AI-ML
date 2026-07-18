"""LlamaIndex QueryEngineTool example."""

import asyncio

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
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

bank_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="bank_knowledge_base",
    description=(
        "Use this tool to answer questions about bank customers, loan products, "
        "credit scores, and bank risk policies."
    ),
)

agent = ReActAgent(
    tools=[bank_tool],
    llm=llm,
    verbose=True,
)


async def main():
    question = "Who wants a home loan, and what is their credit score?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
