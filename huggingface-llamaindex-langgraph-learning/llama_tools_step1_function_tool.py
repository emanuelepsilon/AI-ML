"""
LlamaIndex Tools - Step 1: FunctionTool.

Goal:
Turn a normal Python function into a LlamaIndex tool.

Important idea:
A tool is a callable ability. Later, an agent can decide when to call it.
For now, we call it manually so the concept is clear.
"""

from llama_index.core.tools import FunctionTool


# ---------------------------------------------------------------------------
# 1. NORMAL PYTHON FUNCTION
# ---------------------------------------------------------------------------
# This is just a regular Python function.
#
# The type hint `location: str` and the docstring matter because LLM agents use
# function names, argument names, type hints, and descriptions to understand
# when/how to call tools.
def get_weather(location: str) -> str:
    """Returns a fake weather report for a city."""
    return f"The weather in {location} is sunny and 22 degrees Celsius."


# ---------------------------------------------------------------------------
# 2. WRAP FUNCTION AS A TOOL
# ---------------------------------------------------------------------------
# FunctionTool.from_defaults(...) turns the Python function into a LlamaIndex
# tool object.
#
# The tool now has:
# - a name
# - a description
# - an input schema
# - a callable function underneath
weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Get the current weather for a given city or location.",
)


# ---------------------------------------------------------------------------
# 3. INSPECT TOOL METADATA
# ---------------------------------------------------------------------------
# Metadata is how an agent understands what the tool is for.
#
# If the name/description are bad, the agent may call the wrong tool or pass
# the wrong arguments.
print("Tool metadata:")
print(f"Name: {weather_tool.metadata.name}")
print(f"Description: {weather_tool.metadata.description}")
print(f"Schema: {weather_tool.metadata.fn_schema_str}")


# ---------------------------------------------------------------------------
# 4. CALL THE TOOL MANUALLY
# ---------------------------------------------------------------------------
# Before giving tools to an agent, it is smart to call them yourself.
#
# This proves:
# - the tool works
# - the arguments are correct
# - the output looks useful
result = weather_tool.call(location="Stockholm")

print("\nTool result:")
print(result)
