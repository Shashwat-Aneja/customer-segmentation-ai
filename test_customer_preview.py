import pandas as pd
from segment import segment_customers


def test_preview_limits_do_not_remove_customers_from_results(tmp_path, capsys):
    source = tmp_path / "customers.csv"
    pd.DataFrame({"total_spent": range(30), "visits": range(30)}).to_csv(source, index=False)
    result = segment_customers(source, n_clusters=2, max_display_rows=3)
    assert len(result) == 30
    assert "Showing 3 of 30 customers" in capsys.readouterr().out
