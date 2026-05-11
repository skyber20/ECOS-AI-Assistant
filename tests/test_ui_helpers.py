import unittest

import pandas as pd

from app import ui


class StreamlitUiHelperTest(unittest.TestCase):
    def test_first_artifact_finds_dataset_output(self) -> None:
        response = {
            "artifacts": [
                {"artifact_type": "research_report", "download_url": "/report"},
                {"artifact_type": "build_output_dataset", "download_url": "/dataset"},
            ],
        }

        artifact = ui._first_artifact(response, ["build_output_dataset"])

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact["download_url"], "/dataset")

    def test_chart_from_dataset_uses_design_visualization_spec(self) -> None:
        dataframe = pd.DataFrame(
            {
                "year": [2020, 2021, 2020, 2021],
                "geo": ["US", "US", "EU", "EU"],
                "inflation": [1.2, 4.7, 0.5, 2.6],
            }
        )
        spec = {
            "id": "V1",
            "title": "Inflation dynamics",
            "chart_type": "line",
            "x_axis": "year",
            "y_axis": "inflation",
            "grouping": "geo",
        }

        chart = ui._chart_from_dataset(dataframe, spec, {"columns": []})

        self.assertIsNotNone(chart)
        self.assertEqual(chart.to_dict()["mark"]["type"], "line")

    def test_chart_from_chart_data_filters_by_visualization_id(self) -> None:
        chart_data = pd.DataFrame(
            {
                "visualization_id": ["V1", "V1", "V2"],
                "title": ["A", "A", "B"],
                "geo": ["US", "US", "US"],
                "year": [2020, 2021, 2020],
                "metric": ["inflation", "inflation", "gdp"],
                "value": [1.2, 4.7, 100],
            }
        )

        chart = ui._chart_from_chart_data(
            chart_data,
            {"id": "V1", "chart_type": "bar", "y_axis": "inflation"},
        )

        self.assertIsNotNone(chart)
        chart_payload = chart.to_dict()
        self.assertEqual(chart_payload["mark"]["type"], "bar")
        self.assertEqual(len(chart_payload["datasets"][next(iter(chart_payload["datasets"]))]), 2)


if __name__ == "__main__":
    unittest.main()
