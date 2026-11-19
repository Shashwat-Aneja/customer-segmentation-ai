import pandas as pd
import pytest
import segment
@pytest.mark.parametrize('column',['total_spent','visits'])
def test_boolean_features_do_not_form_customer_clusters(monkeypatch,column):
    df=pd.DataFrame({'total_spent':[10,20,30],'visits':[1,2,3]});df[column]=[True,False,True]
    monkeypatch.setattr(segment.pd,'read_csv',lambda _:df)
    assert segment.segment_customers('ignored',n_clusters=2) is None
