import json
import os
from pathlib import Path
from dotenv import load_dotenv

from llm import LLM
from query_parser import QueryParser

load_dotenv()

llm = LLM(
    model=os.getenv("MODEL_DEFAULT"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

parser = QueryParser(
    llm=llm,
    prompt_path=Path("prompts/query_parser.md"),
)

result = parser.parse(
    "Собери динамику ВРП Архангельской области за 2015-2024 годы"
)

print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
