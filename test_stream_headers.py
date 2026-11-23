import io,pytest
from segment import _read_csv_input
@pytest.mark.parametrize('factory',[io.StringIO,lambda value:io.BytesIO(value.encode('utf-8'))])
def test_duplicate_stream_headers_are_rejected_before_pandas_renames_them(factory):
    source=factory('amount;amount\n1;999\n')
    with pytest.raises(ValueError,match='unique'):_read_csv_input(source,delimiter=';')
    assert not source.closed and source.tell()==0
@pytest.mark.parametrize('factory',[io.StringIO,lambda value:io.BytesIO(value.encode('utf-8'))])
def test_valid_stream_preserves_quoted_fields_and_remains_open(factory):
    source=factory('\ufeffname;amount\n"first;second";12\n')
    frame=_read_csv_input(source,delimiter=';')
    assert frame['name'].tolist()==['first;second'] and frame['amount'].tolist()==[12]
    assert not source.closed

@pytest.mark.parametrize('factory',[io.StringIO,lambda value:io.BytesIO(value.encode('utf-8'))])
def test_validation_restores_nonzero_stream_position(factory):
    source=factory('skip\namount;amount\n1;2\n');source.seek(5)
    with pytest.raises(ValueError,match='unique'):_read_csv_input(source,delimiter=';')
    assert source.tell()==5 and not source.closed

def test_nonseekable_stream_is_rejected_explicitly():
    class Nonseekable(io.StringIO):
        def seekable(self):return False
    source=Nonseekable('amount\n1\n')
    with pytest.raises(ValueError,match='seekable'):_read_csv_input(source)
    assert not source.closed

@pytest.mark.parametrize('prefix', ['\n', '  \r\n', '\ufeff'])
@pytest.mark.parametrize('factory',[io.StringIO,lambda value:io.BytesIO(value.encode('utf-8'))])
def test_blank_preamble_and_bom_cannot_hide_quoted_duplicate_headers(prefix,factory):
    source=factory(prefix+'"amount",amount\n1,999\n')
    with pytest.raises(ValueError,match='unique'):_read_csv_input(source)
    assert source.tell()==0 and not source.closed

@pytest.mark.parametrize('ending', ['\r', '\r\n', '\n'])
def test_text_stream_supports_universal_record_endings(ending):
    frame=_read_csv_input(io.StringIO(ending.join(['name,amount','first,12','second,20',''])))
    assert frame['amount'].tolist()==[12,20]

def test_multiline_quoted_header_preserves_physical_newlines():
    frame=_read_csv_input(io.StringIO('"long\r\nname",amount\r\nfirst,12\r\n'))
    assert frame.columns.tolist()==['long\r\nname','amount']
