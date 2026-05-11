import tempfile
import unittest
from pathlib import Path

from agent.nsedc_client import (
    DownloadedTable,
    NsedcClient,
    NsedcResource,
    profile_table_file,
)


class NsedcClientTest(unittest.TestCase):
    def test_choose_table_resource_skips_metadata_when_data_resource_exists(self) -> None:
        client = NsedcClient(storage_dir="/tmp/nsedc-test")
        resources = [
            NsedcResource(
                id="metadata",
                name="metadata.csv",
                format="CSV",
                url="https://example.test/metadata.csv",
                position=0,
            ),
            NsedcResource(
                id="main",
                name="poverty.xlsx",
                format="XLSX",
                url="https://example.test/poverty.xlsx",
                position=1,
            ),
        ]

        selected = client.choose_table_resource(resources)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.id, "main")

    def test_profile_csv_builds_small_context_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "year,country,value\n2020,Russia,1.5\n2021,Russia,1.7\n",
                encoding="utf-8",
            )

            profile = profile_table_file(table_path, "CSV", sample_rows=1)
            downloaded = DownloadedTable(
                dataset_name="sample_dataset",
                resource=NsedcResource(
                    id="resource-1",
                    name="table.csv",
                    format="CSV",
                    url="https://example.test/table.csv",
                ),
                local_path=str(table_path),
                profile_path=str(Path(tmpdir) / "profile.json"),
                profile=profile,
                source_url="https://example.test/table.csv",
                downloaded_at="2026-01-01T00:00:00+00:00",
            )

            payload = downloaded.context_payload()

        self.assertEqual(payload["columns"], ["year", "country", "value"])
        self.assertEqual(payload["sample_rows"], [{"year": "2020", "country": "Russia", "value": "1.5"}])
        self.assertIn("local_path", payload)
        self.assertNotIn("2021,Russia,1.7", str(payload))


if __name__ == "__main__":
    unittest.main()
