import json
import os
import re
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from repomind.observability.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"


def _load_configs() -> dict:
    """Load MCP server configs and resolve ${VAR} placeholders from env."""
    if not CONFIG_PATH.exists():
        logger.warning(f"MCP config not found at {CONFIG_PATH}")
        return {}

    raw_text = CONFIG_PATH.read_text()
    resolved = re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.getenv(m.group(1), ""),
        raw_text,
    )
    data = json.loads(resolved)
    return data.get("mcpServers", {})


async def get_mcp_tools() -> list:
    """Connect to all configured MCP servers and return their tools."""
    configs = _load_configs()
    if not configs:
        return []

    logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
    client = MultiServerMCPClient(configs)
    tools = await client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    return tools
