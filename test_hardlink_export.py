import os
import segment


def test_output_hardlink_to_input_is_rejected_before_model_fit(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("total_spent,visits\n10,1\n20,2\n")
    output = tmp_path / "alias.csv"
    os.link(source, output)
    def fail(*args, **kwargs):
        raise AssertionError("must reject alias before fitting")
    monkeypatch.setattr(segment, "segment_customers", fail)
    assert segment.main([str(source), "--output", str(output)]) == 1
    assert output.read_bytes() == source.read_bytes()
