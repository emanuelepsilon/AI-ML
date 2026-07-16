import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["server.py"],
)


def print_menu() -> None:
    print()
    print("Choose:")
    print("1. List tools")
    print("2. Add numbers")
    print("3. Save note")
    print("4. Read note resource")
    print("5. Get beginner explanation prompt")
    print("6. Exit")
    print()


def display_result(result) -> None:
    if hasattr(result, "content"):
        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)
            else:
                print(item)
        return

    print(result)


async def main() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to Practice MCP Server.")

            while True:
                print_menu()
                choice = input("> ").strip()

                if choice == "1":
                    tools = await session.list_tools()
                    print("Tools:")
                    for tool in tools.tools:
                        print(f"- {tool.name}: {tool.description}")

                elif choice == "2":
                    a = int(input("a: ").strip())
                    b = int(input("b: ").strip())
                    result = await session.call_tool(
                        "add_numbers",
                        {"a": a, "b": b},
                    )
                    print("Result:")
                    display_result(result)

                elif choice == "3":
                    title = input("title: ").strip()
                    content = input("content: ").strip()
                    result = await session.call_tool(
                        "save_note",
                        {"title": title, "content": content},
                    )
                    print("Result:")
                    display_result(result)

                elif choice == "4":
                    title = input("title: ").strip()
                    result = await session.read_resource(f"notes://{title}")
                    print("Resource:")
                    for item in result.contents:
                        if hasattr(item, "text"):
                            print(item.text)
                        else:
                            print(item)

                elif choice == "5":
                    topic = input("topic: ").strip()
                    result = await session.get_prompt(
                        "explain_like_beginner",
                        {"topic": topic},
                    )
                    print("Prompt:")
                    for message in result.messages:
                        content = message.content
                        if hasattr(content, "text"):
                            print(content.text)
                        else:
                            print(content)

                elif choice == "6":
                    print("Bye.")
                    return

                else:
                    print("Pick 1-6.")


if __name__ == "__main__":
    asyncio.run(main())
