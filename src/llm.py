import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv()


# ── OpenRouter ──────────────────────────────────────────────

@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str = field(default_factory=lambda: os.environ["OPENROUTER_API_KEY"])
    model: str = field(default_factory=lambda: os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"))
    site_url: str = field(default_factory=lambda: os.environ.get("OPENROUTER_SITE_URL", ""))
    app_name: str = field(default_factory=lambda: os.environ.get("OPENROUTER_APP_NAME", ""))


def openrouter_chat_model(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    config: OpenRouterConfig | None = None,
) -> ChatOpenAI:
    cfg = config or OpenRouterConfig()
    return ChatOpenAI(
        model=cfg.model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=cfg.api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": cfg.site_url,
            "X-Title": cfg.app_name,
        },
    )


# ── Bedrock ─────────────────────────────────────────────────

@dataclass(frozen=True)
class BedrockConfig:
    model: str = field(default_factory=lambda: os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"))
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    profile_name: str | None = field(default_factory=lambda: os.environ.get("AWS_PROFILE", "babayaga"))


def bedrock_chat_model(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    config: BedrockConfig | None = None,
) -> ChatBedrock:
    cfg = config or BedrockConfig()
    kwargs = dict(
        model=cfg.model,
        temperature=temperature,
        max_tokens=max_tokens,
        region_name=cfg.region_name,
    )
    if cfg.profile_name:
        kwargs["credentials_profile_name"] = cfg.profile_name
    return ChatBedrock(**kwargs)


# ── Groq ────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroqConfig:
    api_key: str = field(default_factory=lambda: os.environ["GROQ_API_KEY"])
    model: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))


def groq_chat_model(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    config: GroqConfig | None = None,
) -> ChatGroq:
    cfg = config or GroqConfig()
    return ChatGroq(
        model=cfg.model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=cfg.api_key,
    )


if __name__ == "__main__":
    import time

    prompt = "What is the capital of India? Answer in one word."

    providers = []

    if os.environ.get("OPENROUTER_API_KEY"):
        providers.append(("OpenRouter", openrouter_chat_model()))
    try:
        providers.append(("Bedrock", bedrock_chat_model()))
    except Exception:
        pass
    if os.environ.get("GROQ_API_KEY"):
        providers.append(("Groq", groq_chat_model()))

    if not providers:
        print("No API keys found. Set OPENROUTER_API_KEY, AWS_PROFILE/Bedrock, or GROQ_API_KEY.")
        exit(1)

    for name, model in providers:
        try:
            start = time.time()
            response = model.invoke(prompt)
            elapsed = time.time() - start
            print(f"{name:12s} → {response.content.strip():30s}  ({elapsed:.2f}s)")
        except Exception as e:
            print(f"{name:12s} → ERROR: {e}")
