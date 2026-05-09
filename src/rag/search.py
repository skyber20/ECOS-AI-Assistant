import argparse
import json
from pathlib import Path

from .index import RagIndex
from .paths import INDEX_PATH


def main():
    parser = argparse.ArgumentParser(description="Найти релевантные показатели в RAG-индексе")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--limit", default=10, type=int)
    parser.add_argument("--source", choices=["world_bank", "fedstat"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not Path(args.index).exists():
        raise SystemExit(f"Индекс не найден: {args.index}. Сначала запустите python -m src.rag.build_index")

    results = RagIndex(args.index).search(
        " ".join(args.query),
        limit=args.limit,
        source=args.source,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for number, item in enumerate(results, 1):
        citation = item["citation"]
        print(f"{number}. {item['title']}")
        print(f"   score: {item['score']}")
        print(f"   источник: {citation['source']}")
        print(f"   id: {citation['indicator_id']}")
        print(f"   metadata: {citation['metadata_path']}")
        print(f"   data: {citation['data_path']}")
        if citation.get("url"):
            print(f"   url: {citation['url']}")


if __name__ == "__main__":
    main()
