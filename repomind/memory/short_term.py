import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from repomind.config import load_config
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()

MAX_MESSAGES = 4  # keep only the last 4 messages (2 turns)


def get_checkpointer() -> SqliteSaver:
    db_path = config["memory"]["db_path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using SQLite checkpointer at {db_path}")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
