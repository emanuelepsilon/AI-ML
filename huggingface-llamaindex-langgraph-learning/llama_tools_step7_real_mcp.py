"""
LlamaIndex Tools - Step 7: Real MCP

Goal:
Use an actual MCP server, not just a normal Python function.

There are two files:

    1. mcp_bank_server.py
       The MCP server. It exposes bank tools over the MCP protocol.

    2. llama_tools_step7_real_mcp.py
       The MCP client + LlamaIndex agent. It starts/connects to the MCP server,
       discovers the tools, converts them to LlamaIndex tools, and lets the
       agent call them.

Important difference from earlier examples:

Earlier:
    agent imports Python function directly

MCP:
    agent -> MCP client -> MCP protocol -> separate MCP server process -> tool

That is why MCP matters for real apps: the tools can live outside your agent.
"""

import asyncio
import sys
from pathlib import Path

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


# ---------------------------------------------------------------------------
# 1. FIND THE SERVER FILE
# ---------------------------------------------------------------------------
# We start the MCP server as a separate Python process.
#
# sys.executable points to the Python running this client, which should be:
#
#     .\\.venv\\Scripts\\python.exe
#
# That keeps server and client using the same venv/packages.
PROJECT_DIR = Path(__file__).parent
SERVER_FILE = PROJECT_DIR / "mcp_bank_server.py"


# ---------------------------------------------------------------------------
# 2. LOCAL OLLAMA LLM
# ---------------------------------------------------------------------------
# The LLM is still local.
#
# MCP does not replace the LLM.
# MCP replaces how tools are exposed/connectable.
llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 3. MCP CLIENT
# ---------------------------------------------------------------------------
# BasicMCPClient starts/connects to an MCP server.
#
# Here we use stdio transport:
#
#     client starts: python mcp_bank_server.py
#     client talks to server through stdin/stdout
#
# The server process exposes tools like:
# - list_customers
# - get_customer_profile
# - get_loan_policy
# - compare_customer_to_policy
mcp_client = BasicMCPClient(
    command_or_url=sys.executable,
    args=[str(SERVER_FILE)],
    timeout=120,
)


# ---------------------------------------------------------------------------
# 4. MCP TOOL SPEC
# ---------------------------------------------------------------------------
# McpToolSpec asks the MCP server:
#
#     "What tools do you provide?"
#
# Then it converts those MCP tools into LlamaIndex FunctionTool objects.
mcp_tool_spec = McpToolSpec(client=mcp_client)


# ---------------------------------------------------------------------------
# 5. AGENT
# ---------------------------------------------------------------------------
# We create the agent inside async main() because fetching MCP tools is async.
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
