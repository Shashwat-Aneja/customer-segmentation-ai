import argparse
import json
import os
import tempfile
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


def select_cluster_count(features, max_clusters=6):
    """Select a bounded candidate count using silhouette quality on scaled features."""
    if not isinstance(max_clusters, int) or isinstance(max_clusters, bool) or not 2 <= max_clusters <= 10:
        raise ValueError("max_clusters must be an integer from 2 to 10.")
    values = StandardScaler().fit_transform(features)
    upper = min(max_clusters, len(features) - 1, len(features.drop_duplicates()))
    if upper < 2:
        raise ValueError("Automatic selection requires at least three rows and two distinct profiles.")
    best_count, best_score = None, -float("inf")
    for count in range(2, upper + 1):
        labels = KMeans(n_clusters=count, random_state=42, n_init=10).fit_predict(values)
        score = segmentation_silhouette_score(values, labels)
        if score is not None and score > best_score:
            best_count, best_score = count, score
    if best_count is None:
        raise ValueError("No usable automatic segmentation was found.")
    return best_count


def segment_customers(csv_file, n_clusters=3, auto_clusters=False):
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
        if (features["total_spent"] < 0).any():
            raise ValueError("Customer spending must be non-negative.")
        if (features["visits"] < 0).any() or not np.equal(
            features["visits"], np.floor(features["visits"])
        ).all():
            raise ValueError("Customer visits must be non-negative whole numbers.")
        if auto_clusters:
            n_clusters = select_cluster_count(features)
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
        df.attrs["cluster_count"] = n_clusters

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



def export_segments_csv_atomic(result, output_path):
    """Write a complete segmentation export before replacing the destination."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(descriptor)

    try:
        result.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass

def export_segment_summary(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"segments": summarize_segments(result).to_dict(orient="records"),
               "silhouette_score": result.attrs.get("silhouette_score")}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Segment customers from a CSV file.")
    parser.add_argument("csv_file", help="CSV containing total_spent and visits columns")
    cluster_group = parser.add_mutually_exclusive_group()
    cluster_group.add_argument(
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
    parser.add_argument("--summary-output", help="Atomically export segment profiles and silhouette score as JSON")
    cluster_group.add_argument("--auto-clusters", action="store_true", help="Select 2–6 clusters by silhouette score")
    args = parser.parse_args(argv)
    source_path = Path(args.csv_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if output_path is not None:
        same_file = output_path == source_path
        if not same_file and output_path.exists():
            try:
                same_file = output_path.samefile(source_path)
            except OSError:
                same_file = True
        if same_file:
            print("❌ Output CSV must be different from the source file.")
            return 1

    summary_path = Path(args.summary_output).expanduser().resolve() if args.summary_output else None
    if summary_path is not None:
        protected = [source_path] + ([output_path] if output_path is not None else [])
        if any(summary_path == path or (summary_path.exists() and path.exists() and summary_path.samefile(path)) for path in protected):
            print("❌ Summary output must differ from the source and customer export.")
            return 1
    options = {"auto_clusters": True} if args.auto_clusters else {}
    result = segment_customers(args.csv_file, n_clusters=args.clusters, **options)
    if result is None:
        return 1
    if output_path is not None:
        try:
            export_segments_csv_atomic(result, output_path)
        except OSError as exc:
            print(f"❌ Could not write segmentation CSV: {exc}")
            return 1
    if summary_path is not None:
        try:
            export_segment_summary(result, summary_path)
        except (OSError, ValueError) as exc:
            print(f"❌ Could not write segment summary: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
