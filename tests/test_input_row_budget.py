from io import StringIO
import pytest
from segment import _read_csv_input


def test_raw_csv_budget_at_limit_and_overflow():
    assert len(_read_csv_input(StringIO("a,b\n1,2\n3,4\n"), max_rows=2)) == 2
    with pytest.raises(ValueError, match="row limit"):
        _read_csv_input(StringIO("a,b\n1,2\n3,4\n5,6\n"), max_rows=2)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 10_000_001])
def test_invalid_row_budget_precedes_source_open(limit):
    with pytest.raises(ValueError, match="max_rows"):
        _read_csv_input("does-not-exist.csv", max_rows=limit)
