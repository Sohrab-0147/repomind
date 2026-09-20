from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from repomind.config import load_config
from repomind.context.indexers.code_parser import get_source_files, parse_file
from repomind.llm.factory import get_embedder
from repomind.observability.logger import get_logger

logger = get_logger(__name__)
config = load_config()


def get_client() -> QdrantClient:
    return QdrantClient(
        url=config["vector_store"]["url"],
        api_key=config["vector_store"]["api_key"],
    )


def index_codebase(repo_path: str, force_reindex: bool = False) -> QdrantVectorStore:
    """
    Parse source files, embed each chunk, store in Qdrant.
    Skips indexing if collection already has data (unless force_reindex=True).
    """
    collection_name = config["vector_store"]["collection_name"]
    url = config["vector_store"]["url"]
    api_key = config["vector_store"]["api_key"]

    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    # If collection already has data, reuse it
    if collection_name in existing and not force_reindex:
        info = client.get_collection(collection_name)
        if info.points_count > 0:
            logger.info(f"Loaded existing index: {info.points_count} chunks")
            embedder = get_embedder()
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedder,
            )

    if force_reindex and collection_name in existing:
        logger.info(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    embedder = get_embedder()
    logger.info(f"Starting indexing of {repo_path}")

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

    logger.info(f"Embedding {len(docs)} chunks... (this may take 1-2 minutes)")

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedder,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        batch_size=32,
    )

    logger.info(f"Indexing complete. Total chunks: {len(docs)}")
    return vector_store
