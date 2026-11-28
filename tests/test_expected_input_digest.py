from io import BytesIO
import hashlib,gzip
import pytest
from segment import _read_csv_input

def test_digest_matches_snapshot_including_compressed_inputs(tmp_path):
    raw=b'a,b\n1,2\n';digest=hashlib.sha256(raw).hexdigest()
    source=BytesIO(raw)
    assert len(_read_csv_input(source,expected_sha256=digest.upper()))==1 and source.tell()==0
    path=tmp_path/'a.csv.gz';path.write_bytes(gzip.compress(raw))
    assert len(_read_csv_input(path,expected_sha256=digest))==1

def test_wrong_digest_fails_before_csv_parsing(monkeypatch):
    import segment
    monkeypatch.setattr(segment.pd,'read_csv',lambda *a,**kw:pytest.fail('parser must not run'))
    with pytest.raises(ValueError,match='SHA-256'):_read_csv_input(BytesIO(b'a,b\n1,2\n'),expected_sha256='0'*64)

def test_malformed_digest_precedes_source_access():
    with pytest.raises(ValueError,match='hexadecimal'):_read_csv_input('missing',expected_sha256='xyz')


def test_short_reads_cannot_validate_a_prefix_digest():
    raw=b'a,b\n1,2\n3,4\n'
    class Short(BytesIO):
        def read(self,n=-1):return super().read(min(n,8) if n>=0 else 8)
    source=Short(raw)
    with pytest.raises(ValueError,match='SHA-256'):
        _read_csv_input(source,expected_sha256=hashlib.sha256(raw[:8]).hexdigest())
    assert source.tell()==0
    assert len(_read_csv_input(source,expected_sha256=hashlib.sha256(raw).hexdigest()))==2
