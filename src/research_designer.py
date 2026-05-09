# research_designer.py
import json
from pathlib import Path
from llm import LLM
from schemas.research import FormalizedQuery
from schemas.research_design import ResearchDesign


class ResearchDesigner:
    def __init__(self, llm: LLM, prompt_path: Path):
        self.llm = llm
        self.prompt = prompt_path.read_text(encoding="utf-8")
    
    def design(self, formalized_query: FormalizedQuery) -> ResearchDesign:
        schema_json = json.dumps(
            ResearchDesign.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        
        query_context = formalized_query.model_dump_json(
            indent=2,
            exclude={"ambiguities", "assumptions"},
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.prompt}\n\n"
                    f"=== ФОРМАЛИЗОВАННЫЙ ЗАПРОС ===\n"
                    f"{query_context}\n\n"
                    f"=== JSON SCHEMA ДЛЯ ОТВЕТА ===\n"
                    f"{schema_json}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Разработай дизайн исследования. "
                    f"Задача: {formalized_query.task_type}. "
                    f"Цель: {formalized_query.goal}"
                ),
            },
        ]
        
        return self.llm.structured_chat(messages, ResearchDesign)
