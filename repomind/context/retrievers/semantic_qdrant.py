from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document
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
_sparse_cache = None
_reranker_cache = None


def _get_embedder_cached():
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = get_embedder()
    return _embedder_cache


def _get_sparse_cached():
    global _sparse_cache
    if _sparse_cache is None:
        _sparse_cache = FastEmbedSparse(model_name="Qdrant/bm25")
    return _sparse_cache


def _get_reranker_cached():
    global _reranker_cache
    if _reranker_cache is None:
        logger.info("Loading FlashRank cross-encoder reranker")
        _reranker_cache = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2", top_n=5
        )
    return _reranker_cache


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
            embedding=_get_embedder_cached(),
            sparse_embedding=_get_sparse_cached(),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )
    return _vector_store


def retrieve(query: str, k: int = 20, top_n: int = 5) -> list[dict]:
    """Hybrid retrieve (k=20) → cross-encoder rerank → top_n chunks."""
    logger.info(f"Hybrid retrieve k={k} for: {query}")

    raw = _get_store().similarity_search_with_score(query, k=k)
    docs = [
        Document(page_content=d.page_content, metadata=d.metadata)
        for d, _ in raw
    ]

    logger.info(f"Reranking {len(docs)} → top {top_n}")
    reranked = _get_reranker_cached().compress_documents(docs, query)

    chunks = []
    for doc in reranked[:top_n]:
        meta = doc.metadata
        chunks.append({
            "content": doc.page_content,
            "source": meta.get("source", "unknown"),
            "name": meta.get("name", "unknown"),
            "type": meta.get("type", "unknown"),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "distance": 0.0,
        })

    logger.info(f"Returned {len(chunks)} reranked chunks")
    return chunks
