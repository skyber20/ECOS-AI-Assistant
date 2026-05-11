import unittest
from pathlib import Path

from agent.dataset_search_planner import (
    DatasetSearchColumnRequirement,
    DatasetSearchRequest,
)
from agent.nsedc_client import NsedcResource, resource_from_ckan
from agent.nsedc_registry import NsedcDatasetRegistry


class NsedcRegistryTest(unittest.TestCase):
    def test_search_maps_ckan_dataset_to_agent_candidate(self) -> None:
        client = _FakeNsedcClient()
        registry = NsedcDatasetRegistry(client=client, max_candidates=3)
        request = DatasetSearchRequest(
            original_query="бедность пожилых людей в России",
            target_dataset_name="poverty_by_year",
            row_grain="год",
            geography_coverage=["Россия"],
            required_indicators=[
                DatasetSearchColumnRequirement(
                    name="OBS_VALUE",
                    role="indicator",
                    data_type="float",
                    definition="Indicator value.",
                )
            ],
        )

        candidates = registry.search(request)

        self.assertEqual(client.last_query, "poverty_by_year OBS_VALUE Россия")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].dataset_id, "poverty")
        self.assertIn("OBS_VALUE", candidates[0].columns)
        self.assertIn("TIME_PERIOD", candidates[0].join_keys)
        self.assertTrue(any("Full data is not included" in note for note in candidates[0].coverage_notes))


class _FakeNsedcClient:
    base_url = "https://repository.nsedc.ru"
    storage_dir = Path("/tmp/nsedc")

    def __init__(self) -> None:
        self.last_query = None

    def search_datasets(self, query: str, rows: int):
        self.last_query = query
        return {
            "results": [
                {
                    "name": "poverty",
                    "title": "Poverty by year",
                    "notes": (
                        "Dataset fields: `TIME_PERIOD`, `REF_AREA`, `OBS_VALUE`. "
                        "Coverage 2020-2024."
                    ),
                    "metadata_modified": "2026-04-09T11:18:03.098754",
                    "resources": [
                        {
                            "id": "data",
                            "name": "poverty.xlsx",
                            "format": "XLSX",
                            "url": "https://example.test/poverty.xlsx",
                        }
                    ],
                    "tags": [{"name": "poverty"}],
                }
            ]
        }

    def choose_table_resource(self, resources):
        if not resources:
            return None
        resource = resources[0]
        if isinstance(resource, NsedcResource):
            return resource
        return resource_from_ckan(resource)


if __name__ == "__main__":
    unittest.main()
