from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from repomind.config import load_config
from repomind.llm.factory import get_embedder
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_vector_store: QdrantVectorStore | None = None


_embedder_cache = None


def _get_embedder_cached():
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = get_embedder()
    return _embedder_cache

def _get_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        client = QdrantClient(
            url=config["vector_store"]["url"],
            api_key=config["vector_store"]["api_key"],
        )
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=config["vector_store"]["collection_name"],
            embedding=get_embedder_cached(),
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )
    return _vector_store


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Hybrid retrieval: dense + BM25 sparse, fused by Qdrant."""
    logger.info(f"Hybrid retrieve top {k} for: {query}")

    results = _get_store().similarity_search_with_score(query, k=k)

    chunks = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append({
            "content": doc.page_content,
            "source": meta.get("source", "unknown"),
            "name": meta.get("name", "unknown"),
            "type": meta.get("type", "unknown"),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "distance": score,
        })

    logger.info(f"Retrieved {len(chunks)} chunks (hybrid)")
    return chunks
