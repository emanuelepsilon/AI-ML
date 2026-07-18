"""LlamaIndex agent using a function tool."""

import asyncio

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama


def get_weather(location: str) -> str:
    """Returns a fake weather report for a city."""
    return f"The weather in {location} is sunny and 22 degrees Celsius."


weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Get the current weather for a given city or location.",
)

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

agent = ReActAgent(
    tools=[weather_tool],
    llm=llm,
    verbose=True,
)


async def main():
    question = "What is the weather in Stockholm?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
