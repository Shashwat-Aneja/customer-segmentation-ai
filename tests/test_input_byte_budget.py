from io import BytesIO, StringIO
import gzip
import pytest
from segment import _read_csv_input

def test_decompressed_byte_budget_and_stream_ownership(tmp_path):
    raw=b'a,b\n1,2\n'
    for source in [BytesIO(raw),StringIO(raw.decode())]:
        assert len(_read_csv_input(source,max_bytes=len(raw)))==1
        assert source.tell()==0 and not source.closed
    path=tmp_path/'a.csv.gz';path.write_bytes(gzip.compress(raw+b'3,4\n'*100))
    with pytest.raises(ValueError,match='byte limit'):_read_csv_input(path,max_bytes=32)

def test_multibyte_text_is_budgeted_as_encoded_bytes():
    source=StringIO('label,value\ncafé,1\n')
    with pytest.raises(ValueError,match='byte limit'):_read_csv_input(source,max_bytes=len(source.getvalue()))
    assert source.tell()==0

@pytest.mark.parametrize('limit',[True,0,-1,67108865])
def test_bad_byte_budget_precedes_source_open(limit):
    with pytest.raises(ValueError,match='max_bytes'):_read_csv_input('missing',max_bytes=limit)


@pytest.mark.parametrize("text", [False, True])
def test_short_reads_cover_complete_input_and_restore_position(text):
    raw=b'a,b\n1,2\n3,4\n5,6\n'
    Base = StringIO if text else BytesIO
    class Short(Base):
        def read(self,n=-1):
            return super().read(min(n, 8) if n >= 0 else 8)
    source=Short(raw.decode() if text else raw)
    with pytest.raises(ValueError,match='byte limit'):
        _read_csv_input(source,max_bytes=8)
    assert source.tell()==0 and not source.closed
    assert len(_read_csv_input(source,max_bytes=len(raw)))==3
    assert source.tell()==0


def test_truncated_caller_owned_compressed_stream_is_normalized():
    source=gzip.GzipFile(fileobj=BytesIO(gzip.compress(b'a,b\n1,2\n')[:-5]))
    try:
        with pytest.raises(ValueError,match='corrupt or incomplete'):
            _read_csv_input(source,max_bytes=128)
        assert not source.closed and source.tell()==0
    finally:
        source.close()
