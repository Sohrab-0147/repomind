import uuid
from pathlib import Path

from repomind.config import load_config
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()


def _session_file() -> Path:
    """Path to the small file that stores the current session id."""
    db_path = Path(config["memory"]["db_path"])
    return db_path.parent / "current_session"


def get_current_session() -> str:
    """Return the saved session id, or create a new one if none exists."""
    session_file = _session_file()
    if session_file.exists():
        session_id = session_file.read_text().strip()
        if session_id:
            logger.info(f"Resuming session: {session_id}")
            return session_id
    return new_session()


def new_session() -> str:
    """Create a new session id and persist it."""
    session_id = str(uuid.uuid4())
    session_file = _session_file()
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(session_id)
    logger.info(f"Started new session: {session_id}")
    return session_id


def switch_session(session_id: str) -> str:
    """Switch to an existing session id."""
    session_file = _session_file()
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(session_id)
    logger.info(f"Switched to session: {session_id}")
    return session_id
