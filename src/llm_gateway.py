import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://servicesessentials.ibm.com/apis/v3"
DEFAULT_MODEL = "global/anthropic.claude-sonnet-4-5-20250929-v1:0"


def clear_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return api_key


def get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def create_client() -> OpenAI:
    clear_proxy_env()
    return OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
    )


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    max_output_tokens: int = 500,
) -> str:
    client = create_client()
    response = client.responses.create(
        model=model or get_model(),
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text or ""


def generate_json(
    prompt: str,
    *,
    model: str | None = None,
    max_output_tokens: int = 2000,
) -> dict[str, Any]:
    response_text = generate_text(
        prompt,
        model=model,
        max_output_tokens=max_output_tokens,
    )
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
