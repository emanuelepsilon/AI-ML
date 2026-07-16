# Practice MCP Server

This is a tiny MCP server for the Hugging Face MCP course.

It exposes:

- `add_numbers(a, b)`: a simple tool that adds two numbers.
- `save_note(title, content)`: a tool that saves a note in `notes/`.
- `notes://{title}`: a resource that reads a saved note.
- `explain_like_beginner(topic)`: a reusable prompt template.

## Activate the workspace venv

From PowerShell:

```powershell
cd C:\Users\emanu\Desktop\HuggingFace
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of execution policy, use this for the
current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You should then see `(.venv)` at the start of your terminal prompt.

## Install the MCP package

```powershell
python -m pip install -r .\mcp_practice_server\requirements.txt
```

## Run the MCP Inspector

```powershell
cd C:\Users\emanu\Desktop\HuggingFace\mcp_practice_server
mcp dev server.py
```

The inspector should open at:

```text
http://127.0.0.1:6274
```

Use it to see the server's tools, resources, and prompts.

## Run it in the terminal without Node.js

If you do not want to use the browser-based MCP Inspector, run the terminal
client instead:

```powershell
cd C:\Users\emanu\Desktop\HuggingFace
.\.venv\Scripts\Activate.ps1
cd .\mcp_practice_server
python terminal_client.py
```

This starts the MCP server as a subprocess, connects to it over stdio, and lets
you call the tools/resources/prompts from a terminal menu.
