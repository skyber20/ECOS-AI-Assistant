from .index import RagIndex


SOURCE_POLICY = {
    "country_group": ["world_bank", "fedstat"],
    "country": ["world_bank", "fedstat"],
    "federal_subject": ["fedstat", "world_bank"],
    "region": ["fedstat", "world_bank"],
    "city": ["fedstat", "world_bank"],
    "municipality": ["fedstat", "world_bank"],
    "organization": ["fedstat", "world_bank"],
    "custom_area": ["fedstat", "world_bank"],
    "unknown": ["fedstat", "world_bank"],
}


class RagRetriever:
    def __init__(self, index):
        self.index = index if isinstance(index, RagIndex) else RagIndex(index)

    def retrieve(self, query, limit=5):
        query = self.dump(query)
        indicators = query.get("indicators") or [None]
        groups = [
            self.retrieve_indicator(query, indicator, limit)
            for indicator in indicators
        ]

        return {
            "original_query": query.get("original_query"),
            "groups": groups,
            "trace": {
                "mode": "metadata_rag",
                "index": str(self.index.path),
                "documents_are_sources": True,
                "data_files_were_not_read": True,
            },
        }

    def retrieve_indicator(self, query, indicator, limit):
        sources = self.sources_for(query)
        source_groups = [
            {
                "source": source,
                "candidates": self.index.search(
                    self.indicator_query(query, indicator),
                    limit=limit,
                    source=source,
                ),
            }
            for source in sources
        ]

        return {
            "indicator": indicator,
            "source_order": sources,
            "sources": source_groups,
            "candidates": self.unique([
                candidate
                for group in source_groups
                for candidate in group["candidates"]
            ]),
        }

    def indicator_query(self, query, indicator):
        return {
            "original_query": indicator.get("raw") if indicator else query.get("original_query"),
            "goal": indicator.get("display_name") if indicator else query.get("goal"),
            "indicators": [indicator] if indicator else [],
            "geographies": query.get("geographies") or [],
            "time": query.get("time") or {},
            "row_grain": query.get("row_grain") or [],
            "dimensions": query.get("dimensions") or [],
            "assumptions": query.get("assumptions") or [],
        }

    def sources_for(self, query):
        geographies = query.get("geographies") or []
        geo_types = [item.get("type", "unknown") for item in geographies] or ["unknown"]
        ordered = []

        for geo_type in geo_types:
            for source in SOURCE_POLICY.get(geo_type, SOURCE_POLICY["unknown"]):
                if source not in ordered:
                    ordered.append(source)

        return ordered

    def unique(self, candidates):
        seen = set()
        result = []

        for candidate in candidates:
            doc_id = candidate["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result.append(candidate)

        return result

    def dump(self, query):
        return query.model_dump() if hasattr(query, "model_dump") else query
