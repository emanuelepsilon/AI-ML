"""LlamaIndex FunctionTool example."""

from llama_index.core.tools import FunctionTool


def get_weather(location: str) -> str:
    """Returns a fake weather report for a city."""
    return f"The weather in {location} is sunny and 22 degrees Celsius."


weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Get the current weather for a given city or location.",
)

print("Tool metadata:")
print(f"Name: {weather_tool.metadata.name}")
print(f"Description: {weather_tool.metadata.description}")
print(f"Schema: {weather_tool.metadata.fn_schema_str}")

result = weather_tool.call(location="Stockholm")

print("\nTool result:")
print(result)
