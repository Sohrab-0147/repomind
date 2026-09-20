import subprocess
from pathlib import Path

from langchain_core.tools import tool

from repomind.observability.logger import get_logger

logger = get_logger(__name__)

WORKSPACE_ROOT = Path.cwd().resolve()
TIMEOUT_SECONDS = 30

# Patterns that are never allowed to run, regardless of context.
BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "shutdown",
    "reboot",
    "> /dev/sda",
    "chmod -R 777 /",
}


def _is_blocked(command: str) -> bool:
    lowered = command.lower().strip()
    return any(blocked in lowered for blocked in BLOCKED_COMMANDS)


def _format_result(result: subprocess.CompletedProcess) -> str:
    parts = [f"exit_code: {result.returncode}"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr.strip()}")
    return "\n".join(parts)


@tool
def run_command(command: str) -> str:
    """Run a shell command inside the workspace and return its output.
    Times out after 30 seconds. Dangerous commands are blocked."""
    if not command or not command.strip():
        return "Error: command cannot be empty"

    if _is_blocked(command):
        logger.warning(f"Blocked command: {command}")
        return "Error: command is blocked for safety reasons"

    logger.info(f"Running command: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(WORKSPACE_ROOT),
        )
        return _format_result(result)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {TIMEOUT_SECONDS} seconds"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_in_directory(command: str, directory: str) -> str:
    """Run a shell command inside a specific directory of the workspace."""
    if not command or not command.strip():
        return "Error: command cannot be empty"
    if not directory or not directory.strip():
        return "Error: directory cannot be empty"

    if _is_blocked(command):
        logger.warning(f"Blocked command: {command}")
        return "Error: command is blocked for safety reasons"

    target = (WORKSPACE_ROOT / directory).resolve()
    try:
        target.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return f"Error: directory escapes workspace: {directory}"

    if not target.exists():
        return f"Error: directory does not exist: {directory}"
    if not target.is_dir():
        return f"Error: path is not a directory: {directory}"

    logger.info(f"Running command in {directory}: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(target),
        )
        return _format_result(result)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {TIMEOUT_SECONDS} seconds"
    except Exception as e:
        return f"Error: {e}"
