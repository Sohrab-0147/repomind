import os
from pathlib import Path

from langchain_core.tools import tool

from repomind.observability.logger import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
WORKSPACE_ROOT = Path.cwd().resolve()


def _safe_resolve(user_path: str) -> Path:
    """Resolve a path and ensure it stays inside WORKSPACE_ROOT."""
    candidate = (WORKSPACE_ROOT / user_path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {user_path}") from exc
    return candidate


@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file inside the workspace."""
    if not file_path or not file_path.strip():
        return "Error: file path cannot be empty"

    try:
        path = _safe_resolve(file_path)
    except ValueError as e:
        return f"Error: {e}"

    if not path.exists():
        return f"Error: file not found: {file_path}"
    if not path.is_file():
        return f"Error: path is not a file: {file_path}"

    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        return f"Error: file too large ({size} bytes). Max is {MAX_FILE_SIZE_BYTES}"

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text: {file_path}"
    except PermissionError:
        return f"Error: permission denied: {file_path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    if not file_path or not file_path.strip():
        return "Error: file path cannot be empty"

    try:
        path = _safe_resolve(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path.relative_to(WORKSPACE_ROOT)}"
    except ValueError as e:
        return f"Error: {e}"
    except PermissionError:
        return f"Error: permission denied: {file_path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def append_file(file_path: str, content: str) -> str:
    """Append content to an existing file."""
    if not file_path or not file_path.strip():
        return "Error: file path cannot be empty"

    try:
        path = _safe_resolve(file_path)
    except ValueError as e:
        return f"Error: {e}"

    if not path.exists():
        return f"Error: file not found: {file_path}"
    if not path.is_file():
        return f"Error: path is not a file: {file_path}"

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} chars to {file_path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def list_directory(directory: str = ".") -> str:
    """List files and subdirectories inside a directory within the workspace."""
    if not directory or not directory.strip():
        directory = "."

    try:
        path = _safe_resolve(directory)
    except ValueError as e:
        return f"Error: {e}"

    if not path.exists():
        return f"Error: directory not found: {directory}"
    if not path.is_dir():
        return f"Error: path is not a directory: {directory}"

    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"Empty directory: {directory}"
        lines = []
        for entry in entries:
            rel = entry.relative_to(WORKSPACE_ROOT).as_posix()
            kind = "dir " if entry.is_dir() else "file"
            lines.append(f"[{kind}] {rel}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def file_exists(file_path: str) -> str:
    """Check whether a file or directory exists inside the workspace."""
    if not file_path or not file_path.strip():
        return "Error: file path cannot be empty"

    try:
        path = _safe_resolve(file_path)
        return "true" if path.exists() else "false"
    except ValueError as e:
        return f"Error: {e}"
