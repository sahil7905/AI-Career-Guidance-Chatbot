
"""LLM layer: grounded career guidance using Groq."""
import os

from dotenv import load_dotenv
from groq import Groq, NotFoundError, APIStatusError

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACKS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

SYSTEM_PROMPT = """You are an expert AI Career Guidance Counsellor.
Use ONLY the provided CONTEXT for factual claims about careers,
skills and resources. If context is insufficient, say so honestly.
Keep answers concise and practical.

Use this structure:
1. Top 3 Recommended Career Paths
2. Skill Gap Analysis
3. 30-60-90 Day Learning Roadmap
4. Recommended Resources
5. Sources Used

Be encouraging and specific."""


def build_prompt(
    profile: str,
    question: str,
    contexts: list[dict]
) -> str:
    # Limit retrieved context to reduce input tokens.
    selected_contexts = contexts[:3]

    ctx = "\n\n".join(
        f"[{c['source']}] {c['text'][:1200]}"
        for c in selected_contexts
    )

    return (
        f"CONTEXT:\n{ctx}\n\n"
        f"USER PROFILE:\n{profile[:1500]}\n\n"
        f"USER QUESTION:\n{question[:1500]}"
    )


def llm_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def generate(
    profile: str,
    question: str,
    contexts: list[dict],
    history: list[dict] | None = None
) -> str:
    if not llm_available():
        raise RuntimeError("GROQ_API_KEY is missing from the .env file.")

    client = Groq()

    # Keep only recent conversation turns.
    recent_history = (history or [])[-4:]

    # Avoid sending oversized messages or unexpected roles.
    safe_history = []
    for item in recent_history:
        role = item.get("role")
        content = item.get("content", "")

        if role in ("user", "assistant") and isinstance(content, str):
            safe_history.append({
                "role": role,
                "content": content[:1500]
            })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *safe_history,
        {
            "role": "user",
            "content": build_prompt(profile, question, contexts)
        }
    ]

    def call(model: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1200,
            temperature=0.4,
        )
        return response.choices[0].message.content or (
            "Sorry, the model returned an empty response."
        )

    try:
        return call(MODEL)

    except NotFoundError:
        available = {m.id for m in client.models.list().data}

        for name in FALLBACKS:
            if name in available:
                try:
                    return call(name)
                except APIStatusError:
                    continue

        return (
            "The configured Groq model is unavailable or its request "
            "limit was reached. Please try again later or select an "
            "available model."
        )

    except APIStatusError as error:
        if error.status_code == 413 or error.status_code == 429:
            return (
                "Groq request limit reached. The request may be too "
                "large or the API rate limit may have been exceeded. "
                "Please try a fresh chat with a shorter question."
            )
        raise
