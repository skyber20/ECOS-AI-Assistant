import argparse
import sys

from openai import OpenAIError

from intent_parser import IntentParserError, parse_research_intent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a research request into JSON.")
    parser.add_argument("query", nargs="*", help="Natural-language research request.")
    parser.add_argument(
        "--provider",
        choices=["qwen", "yandex"],
        default=None,
        help="Override LLM_PROVIDER from .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override provider model from .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        query = input("Опишите исследовательскую задачу: ").strip()
        print("запустился 1 этап перевода в формальный запрос", file=sys.stderr)

    try:
        intent = parse_research_intent(
            query,
            provider=args.provider,
            model=args.model,
        )
    except (RuntimeError, IntentParserError, ValueError, OpenAIError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(intent.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
