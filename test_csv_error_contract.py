import csv,gzip,bz2,lzma,io,pytest
from segment import _read_csv_input

def test_corrupt_deflate_block_uses_validation_error(tmp_path):
    source=tmp_path/'in.csv.gz';payload=bytearray(gzip.compress(b'amount\n1\n'));payload[10]=(payload[10]&~6)|6;source.write_bytes(payload)
    with pytest.raises(ValueError,match='corrupt or incomplete'):_read_csv_input(source)

@pytest.mark.parametrize('factory',[io.StringIO,lambda text:io.BytesIO(text.encode())])
def test_header_parser_errors_restore_caller_streams(factory):
    source=factory('amount\n1\n');limit=csv.field_size_limit(4)
    try:
        with pytest.raises(ValueError,match='parser limits'):_read_csv_input(source)
        assert source.tell()==0 and not source.closed
    finally:csv.field_size_limit(limit)

@pytest.mark.parametrize('suffix,opener',[('.gz',gzip.open),('.bz2',bz2.open),('.xz',lzma.open)])
def test_truncated_full_read_uses_normal_error_contract(tmp_path,suffix,opener):
    p=tmp_path/('in.csv'+suffix)
    with opener(p,'wt') as f:f.write('amount\n1\n2\n')
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError,match='corrupt or incomplete'):_read_csv_input(p)

def test_truncated_caller_owned_compressed_header_is_normalized_and_kept_open():
    source=gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(b'amount')[:-4]))
    try:
        with pytest.raises(ValueError,match='corrupt or incomplete'):_read_csv_input(source)
        assert source.tell()==0 and not source.closed
    finally:source.close()
