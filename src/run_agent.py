import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from llm import LLM
from query_parser import QueryParser

load_dotenv()

llm = LLM(
    model=os.getenv("MODEL_DEFAULT"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    # project=os.getenv("YANDEX_CLOUD_FOLDER"),
)

parser = QueryParser(
    llm=llm,
    prompt_path=Path("prompts/query_parser.md"),
)

query = " ".join(sys.argv[1:]) or "Собери динамику ВРП Архангельской области за 2015-2024 годы"

result = parser.parse(query)

print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
