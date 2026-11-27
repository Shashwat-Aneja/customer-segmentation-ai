import numpy as np
import pandas as pd
from segment import scale_customer_features, segment_customers

def test_weighted_euclidean_geometry():
    features=pd.DataFrame({"total_spent":[1,2,3],"visits":[1,3,2]})
    ordinary=scale_customer_features(features)
    weighted=scale_customer_features(features,feature_weights=[4,1])
    np.testing.assert_allclose(weighted[:,0],ordinary[:,0]*2)
    np.testing.assert_allclose(weighted[:,1],ordinary[:,1])

def test_weighted_automatic_selection_and_distances(tmp_path):
    src=tmp_path/"a.csv"
    pd.DataFrame({"total_spent":[1,2,3,20,21,22],"visits":[1,2,3,20,21,22]}).to_csv(src,index=False)
    result=segment_customers(src,auto_clusters=True,auto_max_clusters=2,spend_weight=4,include_distances=True)
    assert result.attrs["cluster_count"]==2 and np.isfinite(result.distance_to_centroid).all()
    assert result.attrs["feature_weights"]=={"total_spent":4,"visits":1.0}
    assert segment_customers("missing",spend_weight=0) is None
