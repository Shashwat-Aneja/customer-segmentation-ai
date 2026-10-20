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

    def test_insufficient_distinct_profiles_are_rejected(self):
        path = self.csv_path("total_spent,visits\n10,1\n10,1\n10,1\n")
        self.assertIsNone(segment_customers(path, n_clusters=3))

    def test_non_numeric_features_are_rejected(self):
        path = self.csv_path("total_spent,visits\n10,1\ninvalid,2\n30,3\n")
        self.assertIsNone(segment_customers(path, n_clusters=2))


if __name__ == "__main__":
    unittest.main()
