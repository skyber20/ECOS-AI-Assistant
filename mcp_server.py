import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from hybrid_candidate_retriever import retrieve_candidate_datasets
from intent_parser import ResearchIntent, parse_research_intent

load_dotenv()

mcp = FastMCP(
    "ecos-mathmod-rag",
    instructions=(
        "Локальный MCP для экспериментов с ECOS MATHMOD RAG. "
        "parse_intent разбирает запрос. "
        "search_catalog ищет датасеты по query или intent через hybrid search: embeddings + BM25."
    ),
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
    stateless_http=True,
)


@mcp.tool()
def parse_intent(query: str) -> dict[str, Any]:
    """Разобрать запрос пользователя в ResearchIntent."""
    intent = parse_research_intent(query)
    return {
        "status": "ok",
        "intent": intent.model_dump(mode="json"),
    }


@mcp.tool()
def search_catalog(
    query: str | None = None,
    intent: dict[str, Any] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Найти датасеты по сырому query или intent из parse_intent."""
    if intent is not None:
        user_query: str | ResearchIntent = ResearchIntent.model_validate(intent)
        original_query = user_query.original_query
    elif query and query.strip():
        user_query = query.strip()
        original_query = user_query
    else:
        return {
            "status": "error",
            "query": query or "",
            "results": [],
            "error": "Provide query or intent.",
        }

    results = retrieve_candidate_datasets(
        user_query,
        candidate_top_k=limit,
    )

    return {
        "status": "ok",
        "query": original_query,
        "count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")