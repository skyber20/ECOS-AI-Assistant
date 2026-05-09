import json
from pathlib import Path

from .text import compact, join_parts


class WorldBankAdapter:
    source = "world_bank"
    source_title = "World Bank"

    def __init__(self, data_root):
        self.root = Path(data_root) / "wb" / "wb"

    def iter_documents(self):
        indicators_path = self.root / "indicators.json"
        sources = self.read_sources()

        with indicators_path.open(encoding="utf-8") as file:
            indicators = json.load(file)

        for item in indicators:
            source_id = str((item.get("source") or {}).get("id") or "")
            source_name = compact((item.get("source") or {}).get("value"))
            source_meta = sources.get(source_id, {})
            indicator_id = item.get("id")
            title = item.get("name") or indicator_id
            topics = compact(item.get("topics"))

            payload = {
                "source": self.source,
                "source_title": self.source_title,
                "indicator_id": indicator_id,
                "title": title,
                "unit": item.get("unit") or "",
                "topics": topics,
                "source_id": source_id,
                "source_name": source_name,
                "source_note": item.get("sourceNote") or "",
                "source_organization": item.get("sourceOrganization") or "",
                "source_description": source_meta.get("description") or "",
                "metadata_path": str(indicators_path),
                "source_metadata_path": str(self.root / "metadata" / f"{source_id}.json") if source_id else "",
                "data_path": str(self.root / "parquet" / f"{indicator_id}.parquet"),
                "url": source_meta.get("url") or "",
            }
            id_text = " ".join([indicator_id] * 8)
            title_text = " ".join([title] * 4)

            yield {
                "doc_id": f"world_bank:{indicator_id}",
                "source": self.source,
                "source_id": indicator_id,
                "title": title,
                "text": join_parts(
                    id_text,
                    title_text,
                    item.get("unit"),
                    topics,
                    source_name,
                    item.get("sourceNote"),
                    item.get("sourceOrganization"),
                    source_meta.get("description"),
                ),
                "payload": payload,
            }

    def read_sources(self):
        path = self.root / "sources.json"
        with path.open(encoding="utf-8") as file:
            return {str(item.get("id")): item for item in json.load(file)}


class FedStatAdapter:
    source = "fedstat"
    source_title = "ЕМИСС"

    def __init__(self, data_root):
        self.root = Path(data_root) / "fedstatru" / "fedstatru" / "data"

    def iter_documents(self):
        for path in sorted((self.root / "metadata").glob("*.json")):
            with path.open(encoding="utf-8") as file:
                item = json.load(file)

            props = item.get("props") or item
            code = str(item.get("code") or props.get("code") or path.stem)
            title = item.get("name") or props.get("name") or code

            payload = {
                "source": self.source,
                "source_title": self.source_title,
                "indicator_id": code,
                "title": title,
                "unit": props.get("Единицы измерения") or "",
                "periodicity": props.get("Периодичность и характеристика временного ряда") or "",
                "period": props.get("Длина временного ряда") or props.get("Период действия") or "",
                "last_update": props.get("Последнее обновление данных") or "",
                "dimensions": props.get("Признаки (перечень на базе классификаторов и справочников)") or "",
                "methodology": props.get("Методологические пояснения") or "",
                "formation_source": props.get("Источники и способ формирования показателя") or "",
                "agency": props.get("Ведомство (субъект статистического учета)") or "",
                "department": props.get("Подразделение") or "",
                "placement": props.get("Размещение") or "",
                "comment": props.get("Комментарий") or "",
                "metadata_path": str(path),
                "data_path": str(self.root / "parquet" / f"{code}.parquet"),
                "url": item.get("url") or f"https://www.fedstat.ru/indicator/{code}",
            }

            yield {
                "doc_id": f"fedstat:{code}",
                "source": self.source,
                "source_id": code,
                "title": title,
                "text": join_parts(
                    title,
                    title,
                    title,
                    payload["unit"],
                    payload["periodicity"],
                    payload["period"],
                    payload["dimensions"],
                    payload["methodology"],
                    payload["formation_source"],
                    payload["agency"],
                    payload["department"],
                    payload["placement"],
                    payload["comment"],
                ),
                "payload": payload,
            }


def default_adapters(data_root):
    return [
        WorldBankAdapter(data_root),
        FedStatAdapter(data_root),
    ]
