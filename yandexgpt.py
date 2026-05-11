from parser.intent_parser import SYSTEM_PROMPT, _create_json_completion, create_llm_settings

settings = create_llm_settings(provider="yandex")
content = _create_json_completion(
    settings,
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Покажи динамику ИПЦ России за 2014-2024 годы."},
    ],
)

print(content)
