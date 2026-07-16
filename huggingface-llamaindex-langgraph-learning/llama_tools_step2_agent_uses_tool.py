"""
LlamaIndex Tools - Step 2: Agent uses a FunctionTool.

Goal:
Give a tool to an agent and let the agent decide to call it.

Step 1:
We manually called weather_tool.call(location="Stockholm").

Step 2:
The user asks a normal question, and the agent chooses whether to call
get_weather for itself.
"""

import asyncio

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama


# ---------------------------------------------------------------------------
# 1. NORMAL PYTHON FUNCTION
# ---------------------------------------------------------------------------
# This is the real capability.
#
# The agent itself does not know the weather. It can only answer weather
# questions correctly if it chooses to call this function/tool.
def get_weather(location: str) -> str:
    """Returns a fake weather report for a city."""
    return f"The weather in {location} is sunny and 22 degrees Celsius."


# ---------------------------------------------------------------------------
# 2. WRAP FUNCTION AS A TOOL
# ---------------------------------------------------------------------------
# This gives the function a name, description, and input schema.
#
# The agent reads this metadata to understand:
# - what the tool does
# - when to use it
# - what argument to pass
weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Get the current weather for a given city or location.",
)


# ---------------------------------------------------------------------------
# 3. LOCAL LLM
# ---------------------------------------------------------------------------
# The LLM does the reasoning:
# - read the user question
# - inspect available tools
# - decide whether to call a tool
# - use the tool result to answer
#
# Make sure Ollama is running:
#   ollama serve
llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)


# ---------------------------------------------------------------------------
# 4. AGENT WITH TOOL
# ---------------------------------------------------------------------------
# ReActAgent can reason step by step and call tools.
#
# tools=[weather_tool] means:
# "This agent has access to the get_weather tool."
#
# verbose=True prints the agent's thinking/tool-use steps so you can see what
# happened.
agent = ReActAgent(
    tools=[weather_tool],
    llm=llm,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 5. ASK THE AGENT
# ---------------------------------------------------------------------------
# We ask a normal natural-language question.
#
# The agent should infer:
# - location = Stockholm
# - tool to call = get_weather
async def main():
    question = "What is the weather in Stockholm?"
    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
