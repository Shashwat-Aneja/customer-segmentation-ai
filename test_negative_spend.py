import pandas as pd

from segment import main, segment_customers


def test_negative_spending_is_rejected_before_clustering(tmp_path, capsys):
    source = tmp_path / "customers.csv"
    pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "total_spent": [100, -25, 200],
            "visits": [1, 2, 3],
        }
    ).to_csv(source, index=False)

    assert segment_customers(source) is None
    assert "spending must be non-negative" in capsys.readouterr().out


def test_cli_rejects_negative_spending_without_creating_export(tmp_path, capsys):
    source = tmp_path / "customers.csv"
    output = tmp_path / "segmented.csv"
    pd.DataFrame(
        {
            "total_spent": [100, -1, 300],
            "visits": [1, 2, 3],
        }
    ).to_csv(source, index=False)

    assert main([str(source), "--output", str(output)]) == 1
    assert not output.exists()
    assert "spending must be non-negative" in capsys.readouterr().out
