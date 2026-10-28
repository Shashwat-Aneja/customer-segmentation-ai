import pandas as pd

from segment import main


def test_cli_exports_customer_level_segment_assignments(tmp_path):
    source = tmp_path / "customers.csv"
    output = tmp_path / "results" / "segments.csv"
    source.write_text(
        "customer_id,total_spent,visits\n"
        "c1,10,1\nc2,12,2\nc3,100,8\nc4,120,9\nc5,500,20\n",
        encoding="utf-8",
    )

    assert main([str(source), "--clusters", "2", "--output", str(output)]) == 0

    exported = pd.read_csv(output)
    assert len(exported) == 5
    assert {"customer_id", "segment", "segment_name"}.issubset(exported.columns)
    assert set(exported["segment_name"]).issubset({"Lower Spend", "Higher Spend"})
