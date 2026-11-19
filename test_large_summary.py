import pandas as pd
import pytest
from segment import summarize_segments,assign_segment_names
def test_large_finite_summary_and_names():
    df=pd.DataFrame({'total_spent':[1e308,1e308,1.5e308,1.5e308],'visits':[1,1,2,2],'segment':[0,0,1,1]})
    summary=summarize_segments(df)
    assert summary['average_spend'].tolist()==pytest.approx([1e308,1.5e308])
    assert assign_segment_names(df)['segment_name'].tolist()==['Lower Spend','Lower Spend','Higher Spend','Higher Spend']
