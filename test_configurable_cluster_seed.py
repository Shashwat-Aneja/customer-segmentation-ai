import pandas as pd,pytest
from segment import segment_customers,validate_random_state
def test_explicit_cluster_seed_repeats_customer_assignments(tmp_path):
    p=tmp_path/'customers.csv';pd.DataFrame({'total_spent':[1,2,3,10,11,12],'visits':[1,2,1,8,9,8]}).to_csv(p,index=False)
    first=segment_customers(p,n_clusters=2,random_state=7);second=segment_customers(p,n_clusters=2,random_state=7)
    assert first['segment'].tolist()==second['segment'].tolist()
    assert first.attrs['silhouette_score']==second.attrs['silhouette_score']
@pytest.mark.parametrize('seed',[-1,True,2**32])
def test_invalid_cluster_seed_rejected(seed):
    with pytest.raises(ValueError):validate_random_state(seed)
