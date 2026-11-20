import numpy as np
from segment import segment_customers
def test_finite_large_customer_profiles_can_be_clustered(tmp_path):
    p=tmp_path/'customers.csv';p.write_text('total_spent,visits\n1e308,1\n1.1e308,2\n1.5e308,8\n1.6e308,9\n')
    result=segment_customers(p,n_clusters=2)
    assert result is not None and result['segment'].nunique()==2
    assert np.isfinite(result.attrs['silhouette_score'])
