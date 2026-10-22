import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["total_spent", "visits"]


def summarize_segments(df):
    """Summarize cluster sizes and average customer behavior by segment."""
    required = set(FEATURE_COLUMNS + ["segment"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame must include: {', '.join(sorted(missing))}")
    summary = (
        df.groupby("segment", as_index=False)
        .agg(
            customer_count=("segment", "size"),
            average_spend=("total_spent", "mean"),
            average_visits=("visits", "mean"),
        )
        .sort_values("segment")
        .reset_index(drop=True)
    )
    return summary


def segment_customers(csv_file, n_clusters=3):
    try:
        df = pd.read_csv(csv_file)
        missing = set(FEATURE_COLUMNS).difference(df.columns)
        if missing:
            raise ValueError(f"CSV must include: {', '.join(sorted(missing))}")
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer greater than or equal to 2.")

        df = df.copy()
        for column in FEATURE_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        features = df[FEATURE_COLUMNS]
        if df.empty or features.isna().any().any():
            raise ValueError("Customer features must be numeric and non-empty.")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Customer features must be finite numbers.")
        if len(df) < n_clusters:
            raise ValueError(f"At least {n_clusters} customers are required.")
        if len(features.drop_duplicates()) < n_clusters:
            raise ValueError(f"At least {n_clusters} distinct customer profiles are required.")

        scaled_features = StandardScaler().fit_transform(features)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["segment"] = model.fit_predict(scaled_features)

        columns = (["customer_id"] if "customer_id" in df.columns else []) + FEATURE_COLUMNS + ["segment"]
        print("\n===== Customer Segmentation Results =====")
        print(df[columns].to_string(index=False))
        print("=========================================\n")
        return df
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python segment.py <csv_file>")
        return 2
    return 0 if segment_customers(sys.argv[1]) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
