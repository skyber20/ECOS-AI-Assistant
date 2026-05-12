import unittest

from script_generator import _template_script_content


class ScriptGeneratorTemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        namespace: dict[str, object] = {}
        exec(_template_script_content(), namespace)
        self.namespace = namespace

    def test_geo_matching_uses_intent_synonyms_instead_of_hardcoded_aliases(self) -> None:
        context = {
            "target_structure": {"geography": ["Китай"]},
            "intent": {
                "geography": ["Китай"],
                "objects": [],
                "entities": [],
                "english_query": "GDP of China in 2020",
                "keyword_synonyms": [
                    {
                        "keyword": "Китай",
                        "english_keyword": "China",
                        "synonyms": ["People's Republic of China"],
                    }
                ],
                "time_range": {"start_year": 2020, "end_year": 2020},
            },
        }
        source_row = {
            "country_name": "China",
            "countryiso3code": "CHN",
            "date": 2020,
            "value": 1,
        }

        self.assertTrue(self.namespace["row_matches_target"](source_row, context))
        self.assertEqual(self.namespace["display_geo"](source_row, context), "Китай")

    def test_multi_geo_matching_keeps_labels_separate(self) -> None:
        context = {
            "target_structure": {"geography": ["США", "Китай"]},
            "intent": {
                "geography": ["США", "Китай"],
                "english_query": "GDP of the United States and China",
                "keyword_synonyms": [
                    {
                        "keyword": "США",
                        "english_keyword": "United States",
                        "synonyms": ["United States of America"],
                    },
                    {
                        "keyword": "Китай",
                        "english_keyword": "China",
                        "synonyms": [],
                    },
                ],
                "time_range": {"start_year": 2020, "end_year": 2020},
            },
        }

        self.assertEqual(
            self.namespace["display_geo"](
                {"country_name": "United States of America", "date": 2020, "value": 1},
                context,
            ),
            "США",
        )
        self.assertEqual(
            self.namespace["display_geo"](
                {"country_name": "China", "date": 2020, "value": 1},
                context,
            ),
            "Китай",
        )


if __name__ == "__main__":
    unittest.main()
