import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("YANDEX_API_KEY")
project = os.getenv("YANDEX_PROJECT")
prompt_id = os.getenv("YANDEX_PROMPT_ID")

if not api_key:
    raise RuntimeError("YANDEX_API_KEY is not set. Add it to your .env file.")
if not project:
    raise RuntimeError("YANDEX_PROJECT is not set. Add it to your .env file.")
if not prompt_id:
    raise RuntimeError("YANDEX_PROMPT_ID is not set. Add it to your .env file.")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1"),
    project=project,
)

response = client.responses.create(
    prompt={"id": prompt_id},
    input="hello",
)

print(response.output_text)
