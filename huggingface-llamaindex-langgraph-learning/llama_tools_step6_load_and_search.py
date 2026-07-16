"""
LlamaIndex Tools - Step 6: LoadAndSearchToolSpec

Goal:
Give an agent two related tools:

    1. a LOAD tool
    2. a READ / SEARCH tool

This is close to OnDemandLoaderTool, but the flow is split into two steps.

OnDemandLoaderTool:
    one tool call = load data + index data + search data

LoadAndSearchToolSpec:
    first tool call = load data into an index
    second tool call = search/read the loaded index

That split matters when data should be loaded once, then searched multiple
times without loading it again.
"""

import asyncio

from llama_index.core import Document, Settings, SummaryIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. LOCAL LLM
# ---------------------------------------------------------------------------
# SummaryIndex does not need embeddings, so this example only needs Ollama.
#
# Make sure another terminal has:
#
#     ollama serve
Settings.llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 2. FAKE BANK DOCUMENT SOURCE
# ---------------------------------------------------------------------------
# Imagine this is a slow/large source:
# - a PDF folder
# - company wiki
# - database export
# - SharePoint folder
BANK_MANUALS = {
    "risk": [
        "Risk manual: mortgage applications require credit score review, debt-to-income review, and human review.",
        "Risk manual: debt-to-income above policy limit should be flagged as increased repayment risk.",
        "Risk manual: final credit decisions must be made by authorized human staff.",
    ],
    "loans": [
        "Loan manual: mortgage minimum credit score is 680 and maximum debt-to-income ratio is 0.36.",
        "Loan manual: personal loan minimum credit score is 640 and maximum debt-to-income ratio is 0.40.",
        "Loan manual: small business loan minimum credit score is 660 and maximum debt-to-income ratio is 0.45.",
    ],
}


# ---------------------------------------------------------------------------
# 3. NORMAL LOADER FUNCTION
# ---------------------------------------------------------------------------
# This function only loads documents.
#
# By itself, it does not search or answer.
# LoadAndSearchToolSpec will wrap it and create:
#
# - load_bank_manual(...)
# - read_load_bank_manual(...)
def load_bank_manual(manual_name: str) -> list[Document]:
    """Load one bank manual by name. Valid names: risk, loans."""
    normalized_manual_name = manual_name.lower().strip()
    texts = BANK_MANUALS.get(normalized_manual_name)

    if texts is None:
        available_manuals = ", ".join(BANK_MANUALS)
        return [
            Document(
                text=(
                    f"No manual named {manual_name!r}. "
                    f"Available manuals: {available_manuals}."
                )
            )
        ]

    return [
        Document(
            text=text,
            metadata={"manual_name": normalized_manual_name},
        )
        for text in texts
    ]


# ---------------------------------------------------------------------------
# 4. WRAP LOADER FUNCTION AS A TOOL
# ---------------------------------------------------------------------------
# First we make a normal FunctionTool.
#
# This tool can load docs, but it does not yet provide the read/search tool.
loader_tool = FunctionTool.from_defaults(
    fn=load_bank_manual,
    name="load_bank_manual",
    description=(
        "Loads a bank manual into memory. "
        "manual_name must be 'risk' or 'loans'."
    ),
)


# ---------------------------------------------------------------------------
# 5. TURN LOAD TOOL INTO LOAD + SEARCH TOOL SPEC
# ---------------------------------------------------------------------------
# LoadAndSearchToolSpec takes the loader tool and creates TWO tools:
#
# 1. load_bank_manual(manual_name)
#    Loads docs into an index.
#
# 2. read_load_bank_manual(query)
#    Searches/reads the docs that were loaded.
#
# We use SummaryIndex so the example avoids downloading embedding models.
load_and_search_spec = LoadAndSearchToolSpec.from_defaults(
    tool=loader_tool,
    index_cls=SummaryIndex,
)

bank_manual_tools = load_and_search_spec.to_tool_list()


# ---------------------------------------------------------------------------
# 6. AGENT WITH TWO TOOLS
# ---------------------------------------------------------------------------
# The agent now has:
# - a loader tool
# - a reader/search tool
#
# If it follows the instructions correctly, it should:
# 1. load the correct manual
# 2. read/search the loaded manual
# 3. answer the user
agent = ReActAgent(
    tools=bank_manual_tools,
    llm=Settings.llm,
    verbose=True,
    system_prompt=(
        "You are a polite bank assistant. "
        "For bank manual facts, first load the relevant manual, then read it. "
        "Do not invent policy details."
    ),
)


# ---------------------------------------------------------------------------
# 7. ASK A QUESTION
# ---------------------------------------------------------------------------
# This question should make the agent:
# - call load_bank_manual(manual_name="loans")
# - call read_load_bank_manual(query="mortgage minimum credit score")
async def main():
    question = "In the loans manual, what is the mortgage minimum credit score?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
