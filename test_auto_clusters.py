import pandas as pd
import pytest
from segment import select_cluster_count


def test_automatic_selection_finds_two_separated_customer_groups():
    features = pd.DataFrame({"total_spent": [1, 2, 3, 100, 101, 102], "visits": [1, 2, 1, 20, 21, 20]})
    assert select_cluster_count(features, max_clusters=3) == 2


def test_automatic_selection_rejects_insufficient_profiles():
    with pytest.raises(ValueError, match="three rows"):
        select_cluster_count(pd.DataFrame({"total_spent": [1, 2], "visits": [1, 2]}))
