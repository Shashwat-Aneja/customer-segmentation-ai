import tempfile
import unittest
from pathlib import Path

from segment import segment_customers


class CustomerSegmentationTests(unittest.TestCase):
    def csv_path(self, content):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(content)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    def test_missing_customer_id_is_supported(self):
        path = self.csv_path(
            "total_spent,visits\n10,1\n12,2\n100,8\n120,9\n500,20\n"
        )
        result = segment_customers(path, n_clusters=3)
        self.assertIsNotNone(result)
        self.assertIn("segment", result.columns)
        self.assertEqual(result["segment"].nunique(), 3)

    def test_segmentation_prints_named_segment_profiles(self):
        from contextlib import redirect_stdout
        from io import StringIO

        path = self.csv_path(
            "total_spent,visits\n10,1\n12,2\n100,8\n120,9\n500,20\n"
        )
        output = StringIO()
        with redirect_stdout(output):
            result = segment_customers(path, n_clusters=3)
        self.assertIsNotNone(result)
        self.assertIn("Segment profiles:", output.getvalue())
        self.assertIn("segment_name", result.columns)

    def test_insufficient_distinct_profiles_are_rejected(self):
        path = self.csv_path("total_spent,visits\n10,1\n10,1\n10,1\n")
        self.assertIsNone(segment_customers(path, n_clusters=3))

    def test_non_numeric_features_are_rejected(self):
        path = self.csv_path("total_spent,visits\n10,1\ninvalid,2\n30,3\n")
        self.assertIsNone(segment_customers(path, n_clusters=2))


    def test_segment_summary_aggregates_size_and_behavior(self):
        import pandas as pd
        from segment import summarize_segments

        frame = pd.DataFrame({
            "total_spent": [10, 30, 100],
            "visits": [1, 3, 10],
            "segment": [0, 0, 1],
        })
        summary = summarize_segments(frame).set_index("segment")
        self.assertEqual(summary.loc[0, "customer_count"], 2)
        self.assertAlmostEqual(summary.loc[0, "average_spend"], 20.0)
        self.assertAlmostEqual(summary.loc[1, "average_visits"], 10.0)

    def test_segment_names_follow_average_spend_rank(self):
        import pandas as pd
        from segment import assign_segment_names

        frame = pd.DataFrame({
            "total_spent": [10, 20, 100, 150, 500, 600],
            "visits": [1, 2, 4, 5, 10, 11],
            "segment": [2, 2, 0, 0, 1, 1],
        })
        named = assign_segment_names(frame)
        labels = named.drop_duplicates("segment").set_index("segment")["segment_name"]

        self.assertEqual(labels[2], "Low Spend")
        self.assertEqual(labels[0], "Mid Spend")
        self.assertEqual(labels[1], "High Spend")

    def test_segment_summary_includes_names_when_available(self):
        import pandas as pd
        from segment import summarize_segments

        frame = pd.DataFrame({
            "total_spent": [10, 20, 100, 150, 500, 600],
            "visits": [1, 2, 4, 5, 10, 11],
            "segment": [2, 2, 0, 0, 1, 1],
            "segment_name": ["Low Spend", "Low Spend", "Mid Spend", "Mid Spend", "High Spend", "High Spend"],
        })
        summary = summarize_segments(frame).set_index("segment")
        self.assertEqual(summary.loc[2, "segment_name"], "Low Spend")
        self.assertEqual(summary.loc[1, "segment_name"], "High Spend")

    def test_segment_summary_requires_segment_labels(self):
        import pandas as pd
        from segment import summarize_segments

        with self.assertRaisesRegex(ValueError, "segment"):
            summarize_segments(pd.DataFrame({"total_spent": [10], "visits": [1]}))

if __name__ == "__main__":
    unittest.main()
