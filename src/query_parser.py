import json
from pathlib import Path

from llm import LLM
from schemas.research import FormalizedQuery


class QueryParser:
    def __init__(self, llm: LLM, prompt_path: Path):
        self.llm = llm
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def parse(self, query: str) -> FormalizedQuery:
        schema_json = json.dumps(
            FormalizedQuery.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.prompt}\n\n"
                    f"JSON Schema:\n{schema_json}"
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        return self.llm.structured_chat(messages, FormalizedQuery)
