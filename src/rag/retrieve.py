import argparse
import json
import os

from dotenv import load_dotenv

try:
    from llm import LLM
except ModuleNotFoundError:
    from src.llm import LLM
from .paths import INDEX_PATH
from .paths import PROJECT_ROOT
from .reranker import RagReranker
from .retriever import RagRetriever


def main():
    parser = argparse.ArgumentParser(description="Найти источники по формализованному запросу")
    parser.add_argument("query_json")
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--limit", default=20, type=int)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-limit", default=8, type=int)
    args = parser.parse_args()

    query = json.loads(args.query_json)
    result = RagRetriever(args.index).retrieve(query, limit=args.limit)

    if args.rerank:
        load_dotenv()
        llm = LLM(
            model=os.getenv("MODEL_DEFAULT"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            project=os.getenv("YANDEX_CLOUD_FOLDER"),
        )
        result = {
            "retrieval": result,
            "rerank": RagReranker(
                llm,
                PROJECT_ROOT / "prompts" / "rag_reranker.md",
            ).rerank(query, result, limit=args.rerank_limit),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
