"""LlamaIndex agent using Context for session memory."""

import asyncio

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

agent = ReActAgent(
    tools=[],
    llm=llm,
    verbose=False,
    system_prompt=(
        "You are a concise assistant. If the conversation contains the answer, use it."
    ),
)


async def stateless_example():
    print("\n--- Stateless example: no shared Context ---")

    first_response = await agent.run("My name is Emanu.")
    print("\nFirst response:")
    print(first_response)

    second_response = await agent.run("What is my name?")
    print("\nSecond response:")
    print(second_response)


async def context_memory_example():
    print("\n--- Memory example: shared Context ---")

    ctx = Context(agent)

    first_response = await agent.run("My name is Emanu.", ctx=ctx)
    print("\nFirst response:")
    print(first_response)

    second_response = await agent.run("What is my name?", ctx=ctx)
    print("\nSecond response:")
    print(second_response)


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
