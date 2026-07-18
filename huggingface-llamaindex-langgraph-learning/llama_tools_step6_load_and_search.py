"""LlamaIndex loading and search tool example."""

import asyncio

from llama_index.core import Document, Settings, SummaryIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

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


loader_tool = FunctionTool.from_defaults(
    fn=load_bank_manual,
    name="load_bank_manual",
    description=(
        "Loads a bank manual into memory. manual_name must be 'risk' or 'loans'."
    ),
)

load_and_search_spec = LoadAndSearchToolSpec.from_defaults(
    tool=loader_tool,
    index_cls=SummaryIndex,
)

bank_manual_tools = load_and_search_spec.to_tool_list()

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


async def main():
    question = "In the loans manual, what is the mortgage minimum credit score?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
