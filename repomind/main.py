from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt

from repomind.context.indexers.semantic_qdrant import index_codebase
from repomind.agent.orchestrator import handle_query_stream
from repomind.memory.session import get_current_session, new_session, switch_session
from repomind.observability.logger import get_logger

load_dotenv()

console = Console()
logger = get_logger(__name__)


def run():
    logger.info("Starting RepoMind")
    console.print("\n[bold blue]RepoMind[/bold blue] — RAG-powered code assistant")
    console.print("[dim]Type '/exit' to quit[/dim]\n")

    repo_path = str(Path.cwd())
    console.print(f"[dim]Indexing {repo_path}...[/dim]")
    try:
        index_codebase(repo_path)
        console.print("[green]Ready[/green]\n")
    except Exception as e:
        console.print(f"[red]Indexing failed: {e}[/red]")

    session_id = get_current_session()
    console.print(f"[dim]Session: {session_id}[/dim]\n")

    while True:
        user_input = Prompt.ask("[bold green]>[/bold green]")
        if not user_input.strip():
            continue

        if user_input.lower() in ("/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        elif user_input.startswith("/ask"):
            question = user_input.removeprefix("/ask").strip()
            if not question:
                console.print("[yellow]Usage: /ask <your question>[/yellow]")
                continue

            console.print(f"[dim]Searching for: {question}...[/dim]\n")
            try:
                console.print("[bold]Answer:[/bold] ", end="")
                for token in handle_query_stream(question, session_id):
                    console.print(token, end="", soft_wrap=True)
                console.print("\n")
            except Exception as e:
                logger.error(f"Query failed: {e}")
                console.print(f"\n[red]Error: {e}[/red]\n")

        elif user_input == "/new_session":
            session_id = new_session()
            console.print(f"[green]New session: {session_id}[/green]\n")

        elif user_input.startswith("/switch"):
            target = user_input.removeprefix("/switch").strip()
            if not target:
                console.print("[yellow]Usage: /switch <session_id>[/yellow]")
                continue
            session_id = switch_session(target)
            console.print(f"[green]Switched to: {session_id}[/green]\n")

        elif user_input == "/session":
            console.print(f"[dim]Current session: {session_id}[/dim]\n")

        else:
            console.print("[yellow]Commands:[/yellow]")
            console.print("  /ask <question>       — ask about the codebase")
            console.print("  /new_session          — start a fresh conversation")
            console.print("  /switch <session_id>  — resume a past session")
            console.print("  /session              — show current session id")
            console.print("  /exit                 — quit")


if __name__ == "__main__":
    run()
