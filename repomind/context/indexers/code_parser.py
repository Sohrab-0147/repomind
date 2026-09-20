from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

from repomind.observability.logger import get_logger

logger = get_logger(__name__)

# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
}

TEXT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
ALL_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | TEXT_EXTENSIONS

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

BLOCK_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "arrow_function",
    "class_definition",
    "class_declaration",
    "method_declaration",
    "constructor_declaration",
    "interface_declaration",
    "function_item",
    "func_declaration",
}


@dataclass
class ParsedChunk:
    name: str
    type: str  # "function" | "class" | "block"
    content: str
    source: str
    start_line: int
    end_line: int


def parse_file(filepath: str) -> list[ParsedChunk]:
    ext = Path(filepath).suffix.lower()

    if ext in TEXT_EXTENSIONS:
        lines = Path(filepath).read_text(encoding="utf-8", errors="ignore").splitlines()
        return _sliding_window(lines, filepath)

    language_name = EXTENSION_TO_LANGUAGE.get(ext)
    if not language_name:
        raise ValueError(f"Unsupported file type: {ext}")

    source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    return _parse_with_treesitter(source, filepath, language_name)


def _parse_with_treesitter(source: str, filepath: str, language_name: str) -> list[ParsedChunk]:
    logger.info(f"Parsing {language_name} file: {filepath}")
    parser = get_parser(language_name)
    tree = parser.parse(source.encode())
    lines = source.splitlines()

    chunks: list[ParsedChunk] = []
    _walk(tree.root_node, source, filepath, chunks)

    if not chunks:
        logger.warning(f"No AST blocks found in {filepath}, falling back to sliding window")
        return _sliding_window(lines, filepath)

    logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
    return chunks


def _walk(node, source: str, filepath: str, chunks: list[ParsedChunk]) -> None:
    if node.type in BLOCK_NODE_TYPES:
        name = _extract_name(node, source)
        content = source[node.start_byte : node.end_byte]
        chunk_type = "class" if "class" in node.type else "function"
        chunks.append(
            ParsedChunk(
                name=name,
                type=chunk_type,
                content=content,
                source=filepath,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
        return

    for child in node.children:
        _walk(child, source, filepath, chunks)


def _extract_name(node, source: str) -> str:
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return source[child.start_byte : child.end_byte]
    return node.type


def _sliding_window(lines: list[str], filepath: str) -> list[ParsedChunk]:
    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    chunks: list[ParsedChunk] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i, start in enumerate(range(0, len(lines), step)):
        end = min(start + CHUNK_SIZE, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append(
                ParsedChunk(
                    name=f"chunk{i}",
                    type="block",
                    content=text,
                    source=filepath,
                    start_line=start + 1,
                    end_line=end,
                )
            )
        if end == len(lines):
            break

    return chunks


def get_source_files(repo_path: str, skip_dirs: list[str] | None = None) -> list[str]:
    skip = set(skip_dirs or [".venv", "venv", "__pycache__", ".git", "node_modules", "dist", "build", ".cache", "models"])
    files = [
        str(path)
        for path in Path(repo_path).rglob("*")
        if path.suffix.lower() in ALL_EXTENSIONS
        and not any(part in skip for part in path.parts)
    ]
    logger.info(f"Found {len(files)} source files in {repo_path}")
    return files
