from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from repomind.config import load_config
from repomind.context.indexers.code_parser import get_source_files, parse_file
from repomind.llm.factory import get_embedder
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_SIZE = 384


def get_client() -> QdrantClient:
    return QdrantClient(
        url=config["vector_store"]["url"],
        api_key=config["vector_store"]["api_key"],
    )


def _ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create collection with both dense and sparse vector configs if missing."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        info = client.get_collection(collection_name)
        # Check whether sparse config is present
        has_sparse = bool(getattr(info.config.params, "sparse_vectors", None))
        if has_sparse:
            return
        logger.info(f"Collection {collection_name} lacks sparse config — recreating")
        client.delete_collection(collection_name)

    logger.info(f"Creating hybrid collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=DENSE_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            ),
        },
    )


def index_codebase(repo_path: str, force_reindex: bool = False) -> QdrantVectorStore:
    """Parse source files, embed dense + sparse, store in Qdrant (hybrid mode)."""
    collection_name = config["vector_store"]["collection_name"]
    url = config["vector_store"]["url"]
    api_key = config["vector_store"]["api_key"]

    client = get_client()

    if force_reindex:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            logger.info(f"Force reindex — deleting {collection_name}")
            client.delete_collection(collection_name)

    _ensure_collection(client, collection_name)

    embedder = get_embedder()
    sparse_embedder = FastEmbedSparse(model_name="Qdrant/bm25")

    # If already indexed, load existing store
    info = client.get_collection(collection_name)
    if info.points_count > 0 and not force_reindex:
        logger.info(f"Loaded existing index: {info.points_count} chunks")
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedder,
            sparse_embedding=sparse_embedder,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    logger.info(f"Starting hybrid indexing of {repo_path}")
    files = get_source_files(repo_path)
    docs: list[Document] = []

    for filepath in files:
        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError) as e:
            logger.error(f"Skipping {filepath}: {e}")
            continue

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk.content,
                    metadata={
                        "source": chunk.source,
                        "name": chunk.name,
                        "type": chunk.type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                )
            )

    if not docs:
        raise ValueError(f"No indexable chunks found in {repo_path}")

    logger.info(f"Embedding {len(docs)} chunks (dense + sparse)...")

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=RetrievalMode.HYBRID,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        batch_size=32,
    )

    logger.info(f"Hybrid indexing complete. Total chunks: {len(docs)}")
    return vector_store
