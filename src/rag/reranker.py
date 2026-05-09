import json
from pathlib import Path


class RagReranker:
    def __init__(self, llm, prompt_path):
        self.llm = llm
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")

    def rerank(self, formalized_query, retrieval_result, limit=5):
        payload = {
            "formalized_query": self.dump(formalized_query),
            "retrieval_result": self.compact_result(retrieval_result, limit),
        }
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        raw = self.llm.chat(messages)
        result = self.llm._parse_json(raw)
        return self.attach_sources(result, retrieval_result)

    def compact_result(self, retrieval_result, limit):
        groups = []

        for group in retrieval_result.get("groups", []):
            candidates = []
            for source_group in group.get("sources", []):
                for candidate in source_group.get("candidates", [])[:limit]:
                    payload = candidate["payload"]
                    candidates.append({
                        "doc_id": candidate["doc_id"],
                        "score": candidate["score"],
                        "source": candidate["source"],
                        "indicator_id": candidate["source_id"],
                        "title": candidate["title"],
                        "unit": payload.get("unit"),
                        "source_name": payload.get("source_name") or payload.get("source_title"),
                        "topics": payload.get("topics"),
                        "period": payload.get("period"),
                        "periodicity": payload.get("periodicity"),
                        "dimensions": payload.get("dimensions"),
                        "methodology": self.cut(payload.get("methodology") or payload.get("source_note")),
                    })

            groups.append({
                "indicator": group.get("indicator"),
                "source_order": group.get("source_order"),
                "candidates": candidates,
            })

        return {
            "original_query": retrieval_result.get("original_query"),
            "groups": groups,
        }

    def attach_sources(self, rerank_result, retrieval_result):
        by_doc_id = {
            candidate["doc_id"]: candidate
            for group in retrieval_result.get("groups", [])
            for source_group in group.get("sources", [])
            for candidate in source_group.get("candidates", [])
        }

        for group in rerank_result.get("groups", []):
            for item in group.get("selected", []):
                item["source"] = by_doc_id.get(item.get("doc_id"))

        return rerank_result

    def cut(self, value, limit=700):
        value = str(value or "")
        return value if len(value) <= limit else value[:limit].rstrip()

    def dump(self, value):
        return value.model_dump() if hasattr(value, "model_dump") else value
