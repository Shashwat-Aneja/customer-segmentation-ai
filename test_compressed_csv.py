import gzip,bz2,lzma,pytest
from segment import _read_csv_input
@pytest.mark.parametrize('suffix,opener',[('.gz',gzip.open),('.bz2',bz2.open),('.xz',lzma.open)])
def test_compressed_csv_keeps_header_validation_and_delimiters(tmp_path,suffix,opener):
    path=tmp_path/('input.csv'+suffix)
    with opener(path,'wt',encoding='utf-8') as f:f.write('\ufeff"name";amount\n"first;second";12\n')
    assert _read_csv_input(path,delimiter=';')['name'].tolist()==['first;second']
    with opener(path,'wt',encoding='utf-8') as f:f.write('amount;amount\n1;999\n')
    with pytest.raises(ValueError,match='unique'):_read_csv_input(path,delimiter=';')

@pytest.mark.parametrize('suffix,opener',[('.gz',gzip.open),('.bz2',bz2.open),('.xz',lzma.open)])
def test_truncated_compressed_body_has_a_normalized_validation_error(tmp_path,suffix,opener):
    path=tmp_path/('input.csv'+suffix)
    with opener(path,'wt',encoding='utf-8') as f:f.write('name,amount\nfirst,12\nsecond,20\n')
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError,match='corrupt or incomplete'):_read_csv_input(path)
