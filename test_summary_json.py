import json
import pandas as pd
from segment import export_segment_summary


def test_summary_json_includes_profiles_and_quality_score(tmp_path):
    result = pd.DataFrame({"total_spent": [10, 20, 100], "visits": [1, 2, 9], "segment": [0, 0, 1]})
    result.attrs["silhouette_score"] = 0.8
    output = tmp_path / "summary.json"
    export_segment_summary(result, output)
    payload = json.loads(output.read_text())
    assert payload["silhouette_score"] == 0.8
    assert payload["segments"][0]["customer_count"] == 2
    assert payload["segments"][0]["average_spend"] == 15
