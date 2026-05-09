import json
import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "dummy",
        project: str = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            project=project
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("LLM returned empty response")

        return content

    def structured_chat(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        retries: int = 2,
    ) -> T:
        current_messages = list(messages)

        for _ in range(retries + 1):
            raw = self.chat(current_messages)

            try:
                data = self._parse_json(raw)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as error:
                current_messages.append({
                    "role": "user",
                    "content": (
                        "Ответ не прошел валидацию. "
                        "Верни только валидный JSON без markdown. "
                        f"Ошибка: {error}"
                    ),
                })

        raise ValueError(f"LLM failed to produce valid {schema.__name__}")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        return json.loads(text)
