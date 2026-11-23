from segment import segment_customers
def test_customer_identifiers_keep_leading_zeros(tmp_path):
    p=tmp_path/'customers.csv';p.write_text('id,total_spent,visits\n001,1,1\n002,2,2\n003,10,8\n004,11,9\n')
    result=segment_customers(p,n_clusters=2,id_column='id')
    assert result['id'].tolist()==['001','002','003','004']
def test_duplicate_customer_identifiers_fail_instead_of_weighting_twice(tmp_path):
    p=tmp_path/'customers.csv';p.write_text('id,total_spent,visits\nA,1,1\nA,2,2\nB,10,8\n')
    assert segment_customers(p,n_clusters=2,id_column='id') is None

def test_literal_na_identifiers_are_preserved(tmp_path):
    p=tmp_path/'customers.csv';p.write_text('id,total_spent,visits\nNA,1,1\nNULL,2,2\nN/A,10,8\n000,11,9\n')
    result=segment_customers(p,n_clusters=2,id_column='id')
    assert result['id'].tolist()==['NA','NULL','N/A','000']

def test_missing_numeric_feature_still_rejected_with_literal_id_parsing(tmp_path):
    p=tmp_path/'customers.csv';p.write_text('id,total_spent,visits\nNA,NA,1\nNULL,2,2\nN/A,10,8\n')
    assert segment_customers(p,n_clusters=2,id_column='id') is None
