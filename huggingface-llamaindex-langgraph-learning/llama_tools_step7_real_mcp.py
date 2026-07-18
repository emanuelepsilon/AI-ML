"""LlamaIndex agent connected to banking tools through MCP."""

import asyncio
import sys
from pathlib import Path

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

PROJECT_DIR = Path(__file__).parent
SERVER_FILE = PROJECT_DIR / "mcp_bank_server.py"

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

mcp_client = BasicMCPClient(
    command_or_url=sys.executable,
    args=[str(SERVER_FILE)],
    timeout=120,
)

mcp_tool_spec = McpToolSpec(client=mcp_client)


async def main():
    print("Discovering tools from MCP server...")

    tools = await mcp_tool_spec.to_tool_list_async()

    print("\nMCP tools discovered:")
    for tool in tools:
        print(f"- {tool.metadata.name}: {tool.metadata.description}")

    print("\nCalling one MCP tool directly, before involving the LLM...")
    direct_result = await mcp_client.call_tool(
        "compare_customer_to_policy",
        {
            "customer_id_or_name": "Alice",
            "loan_type": "mortgage",
        },
    )
    print(direct_result.content[0].text)

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are a polite bank assistant. "
            "Use MCP tools for bank facts. "
            "Do not invent customer records or loan policies. "
            "Never make final approval or rejection decisions."
        ),
    )

    question = (
        "Using the MCP bank tools, compare Alice to the mortgage policy. "
        "Do not approve or reject the loan."
    )

    try:
        response = await agent.run(question)
    except ConnectionError:
        print("\nMCP worked, but the LLM agent could not connect to Ollama.")
        print("Start Ollama in another terminal, then run this file again:")
        print("    ollama serve")
        return

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
