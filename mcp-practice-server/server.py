from pathlib import Path

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Practice MCP Server")
NOTES_DIR = Path(__file__).parent / "notes"


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two whole numbers and return the result."""
    return a + b


@mcp.tool()
def save_note(title: str, content: str) -> str:
    """Save a short note in the local notes folder."""
    NOTES_DIR.mkdir(exist_ok=True)
    safe_title = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in title.strip()
    )
    if not safe_title:
        safe_title = "untitled"

    note_path = NOTES_DIR / f"{safe_title}.txt"
    note_path.write_text(content, encoding="utf-8")
    return f"Saved note to {note_path}"


@mcp.resource("notes://{title}")
def read_note(title: str) -> str:
    """Read a saved note by title."""
    safe_title = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in title.strip()
    )
    note_path = NOTES_DIR / f"{safe_title}.txt"
    if not note_path.exists():
        return f"No note found for title: {title}"

    return note_path.read_text(encoding="utf-8")


@mcp.prompt()
def explain_like_beginner(topic: str) -> str:
    """Create a beginner-friendly explanation prompt."""
    return (
        f"Explain {topic} to a beginner. Use simple language, one small example, "
        "and finish with the most important thing to remember."
    )


if __name__ == "__main__":
    mcp.run()
