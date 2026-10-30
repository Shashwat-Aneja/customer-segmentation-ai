import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
    if "segment_name" in df.columns:
        names = df.drop_duplicates("segment").set_index("segment")["segment_name"]
        summary.insert(1, "segment_name", summary["segment"].map(names))
    return summary


def assign_segment_names(df):
    """Attach stable spend-ranked names while retaining original cluster IDs."""
    summary = summarize_segments(df)
    ranked_segments = summary.sort_values(
        ["average_spend", "average_visits", "segment"]
    )["segment"].tolist()

    if len(ranked_segments) == 2:
        names = {ranked_segments[0]: "Lower Spend", ranked_segments[1]: "Higher Spend"}
    elif len(ranked_segments) == 3:
        names = {
            ranked_segments[0]: "Low Spend",
            ranked_segments[1]: "Mid Spend",
            ranked_segments[2]: "High Spend",
        }
    else:
        names = {
            segment: f"Spend Tier {rank + 1}"
            for rank, segment in enumerate(ranked_segments)
        }

    named = df.copy()
    named["segment_name"] = named["segment"].map(names)
    return named


def segmentation_silhouette_score(features, labels):
    """Return a silhouette score when the cluster labels make it mathematically defined."""
    values = np.asarray(features, dtype=float)
    assignments = np.asarray(labels)
    if values.ndim != 2 or assignments.ndim != 1 or len(values) != len(assignments):
        raise ValueError("Features must be a 2D array aligned with one-dimensional cluster labels.")
    if not np.isfinite(values).all():
        raise ValueError("Silhouette features must contain only finite values.")

    cluster_count = np.unique(assignments).size
    if cluster_count < 2 or cluster_count >= len(assignments):
        return None
    return float(silhouette_score(values, assignments))


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
        labels = model.fit_predict(scaled_features)
        df["segment"] = labels
        df = assign_segment_names(df)
        quality_score = segmentation_silhouette_score(scaled_features, labels)
        df.attrs["silhouette_score"] = quality_score

        columns = (["customer_id"] if "customer_id" in df.columns else []) + FEATURE_COLUMNS + ["segment", "segment_name"]
        print("\n===== Customer Segmentation Results =====")
        print(df[columns].to_string(index=False))
        print("\nSegment profiles:")
        print(summarize_segments(df).to_string(index=False))
        if quality_score is None:
            print("\nSilhouette score: unavailable for the resulting cluster labels")
        else:
            print(f"\nSilhouette score: {quality_score:.3f}")
        print("=========================================\n")
        return df
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Segment customers from a CSV file.")
    parser.add_argument("csv_file", help="CSV containing total_spent and visits columns")
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of customer segments to create (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional CSV path for customer-level segment assignments",
    )
    args = parser.parse_args(argv)
    source_path = Path(args.csv_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if output_path is not None and output_path == source_path:
        print("❌ Output CSV must be different from the source file.")
        return 1

    result = segment_customers(args.csv_file, n_clusters=args.clusters)
    if result is None:
        return 1
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)
        except OSError as exc:
            print(f"❌ Could not write segmentation CSV: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
