import argparse
import json

from .adapters import default_adapters
from .index import RagIndex
from .paths import DATA_ROOT, INDEX_PATH


def main():
    parser = argparse.ArgumentParser(description="Построить RAG-индекс по метаданным источников")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--index", default=str(INDEX_PATH))
    args = parser.parse_args()

    index = RagIndex(args.index)
    result = index.build(default_adapters(args.data_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
