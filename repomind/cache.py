from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from repomind.observability.logger import get_logger

logger = get_logger(__name__)

CACHE_PATH = Path(".cache/llm_cache.db")


def init_cache() -> None:
    """Enable SQLite-backed LLM response caching globally."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(CACHE_PATH)))
    logger.info(f"LLM cache enabled at {CACHE_PATH}")
