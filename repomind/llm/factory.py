import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from repomind.config import load_config
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()


def get_llm() -> ChatOpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    base_url = config["llm"]["base_url"]
    api_key = config["llm"]["api_key"]

    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY is missing. Check your .env file.")

    logger.info(f"Using LLM provider: {provider}, model: {model}")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=400,
    )


def get_embedder() -> HuggingFaceEmbeddings:
    """Return a local sentence-transformers embedder. No API key needed."""
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]

    logger.info(f"Using embeddings provider: {provider}, model: {model}")

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
