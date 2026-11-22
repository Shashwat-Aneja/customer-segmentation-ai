import pandas as pd,pytest
from segment import summarize_segments
def test_customer_shares_reconcile_with_counts():
    frame=pd.DataFrame({'segment':[0,0,1],'total_spent':[10,20,30],'visits':[1,2,3]})
    summary=summarize_segments(frame)
    assert summary['customer_share_percent'].tolist()==pytest.approx([200/3,100/3])
    assert summary['customer_share_percent'].sum()==pytest.approx(100)
def test_missing_assignments_do_not_silently_reduce_summary_coverage():
    with pytest.raises(ValueError,match='identifiers'):summarize_segments(pd.DataFrame({'segment':[None],'total_spent':[10],'visits':[1]}))
