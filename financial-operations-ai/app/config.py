import os


def database_url():
    return os.getenv("DATABASE_URL", "sqlite:///./finance_ops.db")


def llm_config():
    return {
        "api_key": os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("LLM_MODEL", ""),
    }

