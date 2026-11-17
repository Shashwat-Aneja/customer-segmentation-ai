from segment import segment_customers


def test_fractional_customer_visit_count_is_rejected(tmp_path, capsys):
    source = tmp_path / "customers.csv"
    source.write_text("total_spent,visits\n10,1.5\n20,2\n30,3\n")
    assert segment_customers(source, n_clusters=2) is None
    assert "whole numbers" in capsys.readouterr().out
