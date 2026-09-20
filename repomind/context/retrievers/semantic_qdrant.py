from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from repomind.config import load_config
from repomind.llm.factory import get_embedder
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()

_vector_store: QdrantVectorStore | None = None


def _get_store() -> QdrantVectorStore:
    """Lazy singleton — reuse the same vector store across queries."""
    global _vector_store
    if _vector_store is None:
        client = QdrantClient(
            url=config["vector_store"]["url"],
            api_key=config["vector_store"]["api_key"],
        )
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=config["vector_store"]["collection_name"],
            embedding=get_embedder(),
        )
    return _vector_store


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Embed the query and return the top-k most similar chunks."""
    logger.info(f"Retrieving top {k} chunks for query: {query}")

    results = _get_store().similarity_search_with_score(query, k=k)

    chunks = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append(
            {
                "content": doc.page_content,
                "source": meta.get("source", "unknown"),
                "name": meta.get("name", "unknown"),
                "type": meta.get("type", "unknown"),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "distance": score,
            }
        )

    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks
