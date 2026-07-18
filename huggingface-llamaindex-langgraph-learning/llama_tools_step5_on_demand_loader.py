"""LlamaIndex on-demand data-loading tool example."""

import asyncio

from llama_index.core import Document, Settings, SummaryIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.readers.base import BaseReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

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


async def main():
    question = "What is the minimum credit score for a mortgage?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
