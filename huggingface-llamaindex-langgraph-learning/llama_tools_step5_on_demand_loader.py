"""
LlamaIndex Tools - Step 5: OnDemandLoaderTool

Goal:
Give an agent a tool that loads data only when the agent needs it.

This is different from our earlier RAG examples:

Earlier RAG:
    load documents -> chunk/embed/index upfront -> query later

OnDemandLoaderTool:
    user asks question -> agent calls tool -> tool loads docs now
    -> tool builds temporary index -> tool searches/answers -> agent responds

This is useful when the data is:
    - large
    - slow to load
    - expensive to fetch
    - only needed sometimes
"""

import asyncio

from llama_index.core import Document, Settings, SummaryIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.readers.base import BaseReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. LOCAL MODELS
# ---------------------------------------------------------------------------
# OnDemandLoaderTool builds an index inside the tool call.
#
# That means it needs:
# - an LLM to answer from the retrieved context
#
# We set the LLM globally through LlamaIndex Settings because the tool creates
# the query engine internally.
#
# For this lesson we use SummaryIndex instead of VectorStoreIndex below.
# SummaryIndex is simpler: it reads over the loaded docs without needing an
# embedding model. That keeps the example fully local/offline except Ollama.
Settings.llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 2. FAKE SLOW / OPTIONAL DATA SOURCE
# ---------------------------------------------------------------------------
# Imagine this dictionary is not normally loaded into memory.
#
# In a real app this could be:
# - files on disk
# - a database
# - SharePoint
# - Google Drive
# - an internal company API
DEPARTMENT_DOCUMENTS = {
    "loans": [
        "Mortgage policy: minimum credit score is 680 and maximum debt-to-income ratio is 0.36.",
        "Personal loan policy: minimum credit score is 640 and maximum debt-to-income ratio is 0.40.",
        "Small business loan policy: minimum credit score is 660 and maximum debt-to-income ratio is 0.45.",
    ],
    "customers": [
        "Customer C001 is Alice Rivera. She wants a mortgage. Her credit score is 720.",
        "Customer C002 is Ben Carter. He wants a personal loan. His credit score is 610.",
        "Customer C003 is Maya Lind. She wants a small business loan. Her credit score is 680.",
    ],
}


# ---------------------------------------------------------------------------
# 3. CUSTOM READER
# ---------------------------------------------------------------------------
# A Reader is responsible for loading documents.
#
# The OnDemandLoaderTool will call this reader only when the tool is used.
#
# The agent/tool passes:
# - query_str: what to ask the loaded data
# - department: which data source to load
#
# Because use_query_str_in_loader=False below, query_str is used for searching,
# not passed into load_data().
class BankDepartmentReader(BaseReader):
    def load_data(self, department: str = "loans"):
        """Load bank documents for one department."""
        normalized_department = department.lower().strip()
        texts = DEPARTMENT_DOCUMENTS.get(normalized_department)

        if texts is None:
            available_departments = ", ".join(DEPARTMENT_DOCUMENTS)
            return [
                Document(
                    text=(
                        f"No department named {department!r}. "
                        f"Available departments: {available_departments}."
                    )
                )
            ]

        return [
            Document(
                text=text,
                metadata={"department": normalized_department},
            )
            for text in texts
        ]


# ---------------------------------------------------------------------------
# 4. ON-DEMAND LOADER TOOL
# ---------------------------------------------------------------------------
# This tool does several things *inside the tool call*:
#
# 1. calls BankDepartmentReader.load_data(...)
# 2. builds a temporary SummaryIndex from the loaded documents
# 3. runs the query against that temporary index
# 4. returns the answer to the agent
#
# Notice the tool is not just "loading".
# It loads + indexes + queries.
bank_department_tool = OnDemandLoaderTool.from_defaults(
    reader=BankDepartmentReader(),
    index_cls=SummaryIndex,
    name="bank_department_loader",
    description=(
        "Loads one bank department's documents on demand, then searches them. "
        "Use this for questions about bank loan policies or customer records. "
        "Inputs: query_str is the question to ask; department is either "
        "'loans' or 'customers'."
    ),
)


# ---------------------------------------------------------------------------
# 5. AGENT WITH ON-DEMAND TOOL
# ---------------------------------------------------------------------------
# The agent does not receive all documents upfront.
#
# It only receives a tool description.
# If the question needs bank data, it should call bank_department_loader.
agent = ReActAgent(
    tools=[bank_department_tool],
    llm=Settings.llm,
    verbose=True,
    system_prompt=(
        "You are a polite bank assistant. "
        "Use tools for bank facts. "
        "Do not invent policies or customer records."
    ),
)


# ---------------------------------------------------------------------------
# 6. ASK A QUESTION
# ---------------------------------------------------------------------------
# This question should make the agent call the on-demand loader.
#
# Expected rough flow:
# question -> agent calls bank_department_loader
#          -> tool loads loan docs
#          -> tool builds temporary index
#          -> tool answers from the loaded docs
#          -> agent gives final response
async def main():
    question = "What is the minimum credit score for a mortgage?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
