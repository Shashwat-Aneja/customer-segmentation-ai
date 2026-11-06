from pathlib import Path

import pandas as pd

import segment
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



def test_failed_export_preserves_previous_report_and_cleans_temporary_file(tmp_path, monkeypatch):
    source = tmp_path / "customers.csv"
    output = tmp_path / "results" / "segments.csv"
    output.parent.mkdir(parents=True)
    output.write_text("previous report\n", encoding="utf-8")
    source.write_text(
        "customer_id,total_spent,visits\n"
        "c1,10,1\nc2,12,2\nc3,100,8\nc4,120,9\nc5,500,20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        segment,
        "segment_customers",
        lambda *args, **kwargs: pd.DataFrame({"customer_id": ["c1"], "segment": [0]}),
    )

    def fail_after_partial_write(_frame, path, index=False):
        Path(path).write_text("partial report\n", encoding="utf-8")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)

    assert main([str(source), "--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "previous report\n"
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
