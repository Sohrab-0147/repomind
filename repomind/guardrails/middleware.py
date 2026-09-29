from langchain.agents.middleware import AgentMiddleware, hook_config

from repomind.observability.logger import get_logger

logger = get_logger(__name__)

BANNED_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your system prompt",
    "you are now",
    "jailbreak",
    "sudo rm",
    "drop table",
    "delete from",
]


class ContentFilterMiddleware(AgentMiddleware):
    """Block prompts containing injection or destructive patterns."""

    def __init__(self, banned: list[str] | None = None):
        super().__init__()
        self.banned = [p.lower() for p in (banned or BANNED_PATTERNS)]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        content = getattr(last, "content", "")
        if not isinstance(content, str):
            return None

        lowered = content.lower()
        for pattern in self.banned:
            if pattern in lowered:
                logger.warning(f"Blocked content pattern: {pattern}")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I can't help with that request. "
                            "Try asking a question about the codebase."
                        ),
                    }],
                    "jump_to": "end",
                }
        return None
