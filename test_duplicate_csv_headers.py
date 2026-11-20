from segment import _reject_duplicate_headers
import pytest
def test_duplicate_headers_are_rejected_instead_of_renamed(tmp_path):
    p=tmp_path/'data.csv';p.write_text('value,value\n1,2\n')
    with pytest.raises(ValueError,match='unique'):_reject_duplicate_headers(p)
def test_bom_header_and_distinct_names_are_supported(tmp_path):
    p=tmp_path/'data.csv';p.write_text('\ufeffvalue,other\n1,2\n',encoding='utf-8')
    _reject_duplicate_headers(p)

def test_duplicate_visit_columns_do_not_reach_clustering(tmp_path):
    from segment import segment_customers
    p=tmp_path/'data.csv';p.write_text('total_spent,visits,visits\n1,1,999\n2,2,999\n3,3,999\n')
    assert segment_customers(p,n_clusters=2) is None
