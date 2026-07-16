# Web Agent

Starter folder for a Gemini-powered agent with web search tools.

## Setup

1. Create a Gemini API key at https://aistudio.google.com/app/apikey
2. Copy `.env.example` to `.env`
3. Replace `your_gemini_api_key_here` with your real key

Your `.env` should look like this:

```env
GEMINI_API_KEY=your_real_key_here
```

Do not commit `.env`; it contains your private API key.

## Run

```powershell
python main.py "Say hello in one sentence"
```

Or run it without arguments and type your question:

```powershell
python main.py
```

The agent uses Gemini with Google Search grounding enabled. When Gemini uses
search, the terminal prints a `Sources:` section with the links returned by the
API.

Google may enforce a separate quota for search-grounded requests. If you see a
quota error, check https://ai.dev/rate-limit or try again after the quota resets.
