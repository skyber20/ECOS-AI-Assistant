from dotenv import load_dotenv

from agent.orchestrator import LangGraphResearchAgent

load_dotenv()

agent = LangGraphResearchAgent(provider="yandex")
intent = agent.parse_intent("Покажи динамику ИПЦ России за 2014-2024 годы.")

print(intent.model_dump_json(indent=2, ensure_ascii=False))
