import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .text import SparseTextEmbedder, idf_from_df, query_to_text


class RagIndex:
    def __init__(self, path, embedder=None):
        self.path = Path(path)
        self.embedder = embedder or SparseTextEmbedder()

    def build(self, adapters):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        df = Counter()
        total = 0

        for document in self.iter_documents(adapters):
            total += 1
            df.update(self.embedder.raw_features(document["text"]).keys())

        idf = idf_from_df(df, total)

        with sqlite3.connect(self.path) as connection:
            self.prepare(connection)
            connection.execute("BEGIN")
            connection.execute("DELETE FROM features")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM idf")
            connection.execute("DELETE FROM meta")
            inserted_features = 0
            connection.executemany(
                "INSERT INTO idf (bucket, value) VALUES (?, ?)",
                idf.items(),
            )

            for document in self.iter_documents(adapters):
                raw = self.embedder.raw_features(document["text"])
                features = self.embedder.weighted_features(raw, idf)
                inserted_features += len(features)

                connection.execute(
                    """
                    INSERT INTO documents
                    (doc_id, source, source_id, title, text, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document["doc_id"],
                        document["source"],
                        document["source_id"],
                        document["title"],
                        document["text"],
                        json.dumps(document["payload"], ensure_ascii=False),
                    ),
                )
                connection.executemany(
                    "INSERT INTO features (bucket, doc_id, weight) VALUES (?, ?, ?)",
                    (
                        (bucket, document["doc_id"], weight)
                        for bucket, weight in features.items()
                    ),
                )

            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("documents", str(total)),
                    ("features", str(inserted_features)),
                    ("buckets", str(self.embedder.buckets)),
                    ("max_features", str(self.embedder.max_features)),
                    ("created_at", str(int(time.time()))),
                ],
            )
            connection.commit()

        return {
            "documents": total,
            "features": inserted_features,
            "seconds": round(time.time() - started, 2),
            "path": str(self.path),
        }

    def search(self, query, limit=10, source=None):
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        text = query_to_text(query)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            raw = self.embedder.raw_features(text)
            idf = self.load_query_idf(connection, raw.keys())
            features = self.embedder.weighted_features(
                raw,
                idf,
            )
            connection.execute("DROP TABLE IF EXISTS query_features")
            connection.execute(
                "CREATE TEMP TABLE query_features (bucket INTEGER PRIMARY KEY, weight REAL)"
            )
            connection.executemany(
                "INSERT INTO query_features (bucket, weight) VALUES (?, ?)",
                features.items(),
            )

            where = "WHERE d.source = ?" if source else ""
            params = [source] if source else []
            params.append(limit)

            rows = connection.execute(
                f"""
                SELECT
                    d.doc_id,
                    d.source,
                    d.source_id,
                    d.title,
                    d.payload,
                    SUM(q.weight * f.weight) AS score
                FROM query_features q
                JOIN features f ON f.bucket = q.bucket
                JOIN documents d ON d.doc_id = f.doc_id
                {where}
                GROUP BY d.doc_id
                ORDER BY score DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [self.row_to_result(row) for row in rows]

    def get(self, doc_id):
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return self.row_to_result(row) if row else None

    def prepare(self, connection):
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS features (
                bucket INTEGER NOT NULL,
                doc_id TEXT NOT NULL,
                weight REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_features_bucket ON features(bucket);
            CREATE INDEX IF NOT EXISTS idx_features_doc_id ON features(doc_id);
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS idf (
                bucket INTEGER PRIMARY KEY,
                value REAL NOT NULL
            );
            """
        )

    def load_query_idf(self, connection, buckets):
        buckets = list(buckets)
        if not buckets:
            return {}
        placeholders = ",".join("?" for _ in buckets)
        rows = connection.execute(
            f"SELECT bucket, value FROM idf WHERE bucket IN ({placeholders})",
            buckets,
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def iter_documents(self, adapters):
        seen = set()
        for adapter in adapters:
            for document in adapter.iter_documents():
                doc_id = document["doc_id"]
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                yield document

    def row_to_result(self, row):
        payload = json.loads(row["payload"])
        return {
            "doc_id": row["doc_id"],
            "source": row["source"],
            "source_id": row["source_id"],
            "title": row["title"],
            "score": round(float(row["score"]), 6) if "score" in row.keys() else None,
            "payload": payload,
            "citation": self.citation(payload),
        }

    def citation(self, payload):
        return {
            "source": payload.get("source_title"),
            "indicator_id": payload.get("indicator_id"),
            "title": payload.get("title"),
            "unit": payload.get("unit"),
            "metadata_path": payload.get("metadata_path"),
            "data_path": payload.get("data_path"),
            "url": payload.get("url"),
        }
