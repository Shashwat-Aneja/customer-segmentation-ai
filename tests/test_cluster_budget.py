import pandas as pd
from segment import segment_customers

def test_cluster_budget_controls_automatic_fit(tmp_path):
    src=tmp_path/"a.csv"
    pd.DataFrame({"total_spent":[1,2,3,20,21,22],"visits":[1,2,3,20,21,22]}).to_csv(src,index=False)
    result=segment_customers(src,auto_clusters=True,auto_max_clusters=2)
    assert result.attrs["cluster_count"]==2
    assert segment_customers("missing",auto_clusters=True,auto_max_clusters=0) is None
