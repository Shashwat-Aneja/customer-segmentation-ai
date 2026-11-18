import numpy as np
import segment


def test_quality_scoring_bounds_rows_and_preserves_rare_clusters(monkeypatch):
    values = np.arange(6000).reshape(3000, 2)
    labels = np.array([0] * 2999 + [1])
    calls = []
    def score(x, y):
        calls.append((x.copy(), y.copy()))
        assert len(x) == 2000
        assert set(y) == {0, 1}
        return 0.5
    monkeypatch.setattr(segment, "silhouette_score", score)
    assert segment.segmentation_silhouette_score(values, labels) == 0.5
    segment.segmentation_silhouette_score(values, labels)
    assert np.array_equal(calls[0][0], calls[1][0])
