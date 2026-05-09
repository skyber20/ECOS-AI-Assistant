import hashlib
import math
import re
import unicodedata
from collections import Counter


TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalize(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.lower().replace("ё", "е")
    return SPACE_RE.sub(" ", text).strip()


def compact(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return SPACE_RE.sub(" ", value).strip()
    if isinstance(value, dict):
        if "value" in value:
            return compact(value.get("value"))
        if "name" in value:
            return compact(value.get("name"))
        return " ".join(compact(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(compact(item) for item in value)
    return str(value)


def join_parts(*parts):
    return "\n".join(part for part in (compact(part) for part in parts) if part)


class SparseTextEmbedder:
    def __init__(self, buckets=262144, max_features=256):
        self.buckets = buckets
        self.max_features = max_features

    def raw_features(self, text):
        tokens = [
            token
            for token in TOKEN_RE.findall(normalize(text))
            if not token.isdigit()
        ]
        counts = Counter()

        for token in tokens:
            counts[self.bucket("w:" + token)] += 4.0
            limit = min(6, len(token) + 1)
            for size in range(3, limit):
                for start in range(0, len(token) - size + 1):
                    counts[self.bucket("c:" + token[start:start + size])] += 1.0

        for left, right in zip(tokens, tokens[1:]):
            counts[self.bucket("b:" + left + " " + right)] += 3.0

        return dict(counts.most_common(self.max_features))

    def weighted_features(self, features, idf):
        weighted = {
            bucket: (1.0 + math.log(value)) * idf.get(bucket, 1.0)
            for bucket, value in features.items()
            if value > 0
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {bucket: value / norm for bucket, value in weighted.items()}

    def bucket(self, feature):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.buckets


def idf_from_df(df, total):
    return {
        bucket: math.log((total + 1.0) / (count + 1.0)) + 1.0
        for bucket, count in df.items()
    }


def query_to_text(query):
    if isinstance(query, str):
        return query
    if hasattr(query, "model_dump"):
        query = query.model_dump()
    if not isinstance(query, dict):
        return str(query)

    parts = [
        query.get("original_query"),
        query.get("goal"),
        " ".join(query.get("row_grain") or []),
        " ".join(query.get("dimensions") or []),
        " ".join(query.get("assumptions") or []),
    ]

    for item in query.get("indicators") or []:
        parts.append(join_parts(
            item.get("raw"),
            item.get("display_name"),
            item.get("canonical_name"),
            item.get("unit"),
            item.get("kind"),
            item.get("aggregation"),
        ))

    for item in query.get("geographies") or []:
        parts.append(join_parts(
            item.get("raw"),
            item.get("name"),
            item.get("type"),
            item.get("country"),
            item.get("role"),
        ))

    time = query.get("time") or {}
    parts.append(join_parts(
        time.get("raw"),
        time.get("start"),
        time.get("end"),
        " ".join(time.get("points") or []),
        time.get("granularity"),
    ))

    return join_parts(*parts)
