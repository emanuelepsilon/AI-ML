"""
LlamaIndex Agents - Step 1: Context memory

Goal:
Show that LlamaIndex agents are stateless by default, but can remember previous
messages when you pass a Context object.

Important idea:

    No Context:
        each agent.run(...) is like a fresh conversation

    Same Context:
        agent.run(..., ctx=ctx) keeps conversation state across calls

This is not long-term database memory.
This is conversation/session memory for one running workflow.
"""

import asyncio

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. LOCAL LLM
# ---------------------------------------------------------------------------
# Make sure Ollama is running in another terminal:
#
#     ollama serve
#
# This is the same local model style we have used in previous lessons.
llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 2. AGENT
# ---------------------------------------------------------------------------
# No tools this time.
#
# We are not testing tool use here.
# We are testing whether the agent remembers prior messages.
agent = ReActAgent(
    tools=[],
    llm=llm,
    verbose=False,
    system_prompt=(
        "You are a concise assistant. "
        "If the conversation contains the answer, use it."
    ),
)


# ---------------------------------------------------------------------------
# 3. STATELESS RUNS
# ---------------------------------------------------------------------------
# These two calls do NOT share a Context.
#
# So the second call should not reliably know the name from the first call.
async def stateless_example():
    print("\n--- Stateless example: no shared Context ---")

    first_response = await agent.run("My name is Emanu.")
    print("\nFirst response:")
    print(first_response)

    second_response = await agent.run("What is my name?")
    print("\nSecond response:")
    print(second_response)


# ---------------------------------------------------------------------------
# 4. MEMORY RUNS WITH CONTEXT
# ---------------------------------------------------------------------------
# Here we create one Context object and reuse it.
#
# This tells LlamaIndex:
# "These calls belong to the same conversation/session."
async def context_memory_example():
    print("\n--- Memory example: shared Context ---")

    ctx = Context(agent)

    first_response = await agent.run("My name is Emanu.", ctx=ctx)
    print("\nFirst response:")
    print(first_response)

    second_response = await agent.run("What is my name?", ctx=ctx)
    print("\nSecond response:")
    print(second_response)


# ---------------------------------------------------------------------------
# 5. RUN BOTH
# ---------------------------------------------------------------------------
# Expected learning:
# - without Context: weak/no memory
# - with Context: remembers the earlier message
async def main():
    try:
        await stateless_example()
        await context_memory_example()
    except ConnectionError:
        print("Could not connect to Ollama.")
        print("Start Ollama in another terminal, then run this file again:")
        print("    ollama serve")


if __name__ == "__main__":
    asyncio.run(main())
