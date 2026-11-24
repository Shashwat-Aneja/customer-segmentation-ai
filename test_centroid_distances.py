import numpy as np,pandas as pd
from segment import segment_customers,scale_customer_features,FEATURE_COLUMNS

def test_centroid_distances_match_scaled_cluster_geometry(tmp_path):
    p=tmp_path/'in.csv';p.write_text('total_spent,visits\n1,1\n2,2\n5,2\n100,9\n110,10\n')
    result=segment_customers(p,n_clusters=2,include_distances=True)
    scaled=scale_customer_features(result[FEATURE_COLUMNS])
    expected=np.empty(len(result))
    for label in result['segment'].unique():
        mask=(result['segment']==label).to_numpy()
        expected[mask]=np.linalg.norm(scaled[mask]-scaled[mask].mean(axis=0),axis=1)
    assert np.allclose(result['distance_to_centroid'],expected)
    assert len(result)==5 and np.isfinite(expected).all()
    assert 'distance_to_centroid' not in segment_customers(p,n_clusters=2).columns
