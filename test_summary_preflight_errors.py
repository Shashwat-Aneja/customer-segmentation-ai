from pathlib import Path
import segment
def test_unreadable_summary_alias_check_fails_before_clustering(tmp_path,monkeypatch):
    source=tmp_path/'source.csv';source.write_text('keep source')
    summary=tmp_path/'summary.json';summary.write_text('keep report')
    def denied(*a):raise OSError('identity unavailable')
    monkeypatch.setattr(Path,'samefile',denied)
    monkeypatch.setattr(segment,'segment_customers',lambda *a,**k:(_ for _ in ()).throw(AssertionError('model should not run')))
    assert segment.main([str(source),'--summary-output',str(summary)])==1
    assert source.read_text()=='keep source' and summary.read_text()=='keep report'
