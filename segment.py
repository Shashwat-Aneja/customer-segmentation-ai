import hashlib
import hmac
import zlib
import gzip
import bz2
import lzma
import io
import csv
import argparse
import math
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


def _snapshot_csv_input(csv_file, max_bytes, encoding):
    """Snapshot bounded decompressed CSV bytes without consuming caller streams."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_bytes must be an integer from 1 to 67108864.")
    def read_complete(source):
        chunks = []
        units = 0
        kind = None
        while units <= max_bytes:
            chunk = source.read(min(65536, max_bytes + 1 - units))
            if not isinstance(chunk, (str, bytes)):
                raise ValueError("CSV stream read must return bytes or text.")
            if kind is not None and type(chunk) is not kind:
                raise ValueError("CSV stream read type must be consistent.")
            kind = type(chunk)
            if not chunk:
                break
            chunks.append(chunk)
            units += len(chunk)
        # Encode text once so codecs with BOMs do not insert one per chunk.
        raw = ("".join(chunks).encode(encoding) if kind is str else b"".join(chunks))
        return raw
    if isinstance(csv_file, (str, os.PathLike)):
        path = Path(csv_file)
        if not path.is_file():
            raise ValueError("Bounded CSV input must be a local regular file or seekable stream.")
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix.lower(), open)
        try:
            with opener(path, "rb") as source:
                raw = read_complete(source)
        except (EOFError, lzma.LZMAError, zlib.error) as exc:
            raise ValueError("CSV input is corrupt or incomplete.") from exc
    else:
        if not all(hasattr(csv_file, name) for name in ("read", "seek", "tell")) or (hasattr(csv_file, "seekable") and not csv_file.seekable()):
            raise ValueError("Bounded CSV streams must be seekable.")
        position = csv_file.tell()
        try:
            raw = read_complete(csv_file)
        except (EOFError, lzma.LZMAError, zlib.error) as exc:
            raise ValueError("CSV input is corrupt or incomplete.") from exc
        finally:
            csv_file.seek(position)
        if isinstance(raw, str):
            raw = raw.encode(encoding)
        if not isinstance(raw, bytes):
            raise ValueError("CSV stream read must return bytes or text.")
    if len(raw) > max_bytes:
        raise ValueError("CSV input exceeds the configured decompressed byte limit.")
    return io.BytesIO(raw)


def _read_csv_input(csv_file, delimiter=",", max_rows=None, encoding="utf-8", max_bytes=None, expected_sha256=None, **options):
    if encoding not in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        raise ValueError("Unsupported input encoding.")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 10_000_000):
        raise ValueError("max_rows must be an integer from 1 to 10000000.")
    if delimiter not in (",", ";", "\t", "|"):
        raise ValueError("Unsupported input delimiter.")
    if max_rows is not None:
        if "nrows" in options:
            raise ValueError("Do not combine nrows and max_rows.")
        options["nrows"] = max_rows + 1
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        csv_file = _snapshot_csv_input(csv_file, max_bytes if max_bytes is not None else 64 * 1024 * 1024, encoding)
        actual = hashlib.sha256(csv_file.getbuffer()).hexdigest()
        if not hmac.compare_digest(actual, expected_sha256.lower()):
            raise ValueError("CSV input SHA-256 does not match the expected digest.")
    elif max_bytes is not None:
        csv_file = _snapshot_csv_input(csv_file, max_bytes, encoding)
    _reject_duplicate_headers(csv_file, delimiter, encoding=encoding)
    try:
        frame = pd.read_csv(csv_file, **({"encoding": encoding} if encoding != "utf-8" else {}), **options, **({"sep": delimiter} if delimiter != "," else {}))
    except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
        raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
    if max_rows is not None and len(frame) > max_rows:
        raise ValueError("CSV input exceeds the configured row limit.")
    return frame


def escape_csv_formulas(frame):
    """Escape formula-like text in a copy; leave numeric cells unchanged."""
    def escape(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    return frame.apply(lambda column: column.map(escape))


def _reject_duplicate_headers(csv_file, delimiter=",", encoding="utf-8"):
    """Inspect original UTF-8 CSV headers without taking ownership of streams."""
    def header_lines(source):
        # csv.reader needs physical lines even for StringIO objects whose default
        # iterator does not recognize CR-only records. Preserve quoted newlines.
        pending = ""
        first = True
        started = False
        while True:
            chars = []
            while True:
                char = pending or source.read(1)
                pending = ""
                if not char:
                    break
                if first:
                    first = False
                    if char == "\ufeff":
                        continue
                chars.append(char)
                if char == "\r":
                    following = source.read(1)
                    if following == "\n":
                        chars.append(following)
                    else:
                        pending = following
                    break
                if char == "\n":
                    break
            if not chars:
                return
            line = "".join(chars)
            if not started and not line.strip():
                continue
            started = True
            yield line

    if isinstance(csv_file, (str, os.PathLike)):
        if not Path(csv_file).is_file():
            return
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(Path(csv_file).suffix, open)
        try:
            with opener(csv_file, mode="rt", encoding=encoding, newline="") as source:
                headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
            raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
    else:
        if not all(hasattr(csv_file, name) for name in ("read", "seek", "tell")) or (hasattr(csv_file, "seekable") and not csv_file.seekable()):
            raise ValueError("CSV streams must be seekable for header validation.")
        position = csv_file.tell()
        binary = isinstance(csv_file.read(0), bytes)
        source = io.TextIOWrapper(csv_file, encoding=encoding, newline="") if binary else csv_file
        try:
            headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
            raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
        finally:
            if binary:
                source.detach()
            csv_file.seek(position)
    if len(headers) != len(set(headers)):
        raise ValueError("CSV column names must be unique; duplicate headers are ambiguous.")


def _stable_group_mean(values):
    numbers = np.asarray(values, dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Segment summaries require finite numeric features.")
    scale = float(np.max(np.abs(numbers))) if numbers.size else 0.0
    return scale * (math.fsum(float(value) for value in numbers / scale) / len(numbers)) if scale else 0.0


def summarize_segments(df):
    """Summarize cluster sizes and average customer behavior by segment."""
    required = set(FEATURE_COLUMNS + ["segment"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame must include: {', '.join(sorted(missing))}")
    if df["segment"].isna().any():
        raise ValueError("Segment identifiers cannot be missing from a summary.")
    summary = (
        df.groupby("segment", as_index=False)
        .agg(
            customer_count=("segment", "size"),
            average_spend=("total_spent", _stable_group_mean),
            average_visits=("visits", _stable_group_mean),
        )
        .sort_values("segment")
        .reset_index(drop=True)
    )
    summary["customer_share_percent"] = summary["customer_count"] / len(df) * 100 if len(df) else 0.0
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


def validate_random_state(value):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        raise ValueError("random_state must be an integer from 0 through 2**32 - 1.")


def segmentation_silhouette_score(features, labels, max_samples=2000, random_state=42):
    """Return a silhouette score when the cluster labels make it mathematically defined."""
    validate_random_state(random_state)
    values = np.asarray(features, dtype=float)
    assignments = np.asarray(labels)
    if values.ndim != 2 or assignments.ndim != 1 or len(values) != len(assignments):
        raise ValueError("Features must be a 2D array aligned with one-dimensional cluster labels.")
    if not np.isfinite(values).all():
        raise ValueError("Silhouette features must contain only finite values.")

    cluster_count = np.unique(assignments).size
    if cluster_count < 2 or cluster_count >= len(assignments):
        return None
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < cluster_count + 1:
        raise ValueError("max_samples must allow at least one sample per cluster plus one.")
    if len(assignments) > max_samples:
        generator = np.random.default_rng(random_state)
        representatives = np.array([generator.choice(np.flatnonzero(assignments == label)) for label in np.unique(assignments)])
        remaining = np.setdiff1d(np.arange(len(assignments)), representatives)
        sampled = np.concatenate([representatives, generator.choice(remaining, max_samples - cluster_count, replace=False)])
        values, assignments = values[sampled], assignments[sampled]
    return float(silhouette_score(values, assignments))


def scale_customer_features(features, feature_weights=None):
    """Normalize magnitude before centering so finite values cannot overflow."""
    scale = features.abs().max().replace(0, 1.0)
    weights = [1.0, 1.0] if feature_weights is None else feature_weights
    if not isinstance(weights, (list, tuple)) or len(weights) != len(FEATURE_COLUMNS) or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 1e-12 <= value <= 1000 for value in weights):
        raise ValueError("Feature weights must be two finite numbers from 1e-12 to 1000.")
    return StandardScaler().fit_transform(features / scale) * np.sqrt(weights)


def select_cluster_count(features, max_clusters=6, random_state=42, feature_weights=None):
    validate_random_state(random_state)
    """Select a bounded candidate count using silhouette quality on scaled features."""
    if not isinstance(max_clusters, int) or isinstance(max_clusters, bool) or not 2 <= max_clusters <= 10:
        raise ValueError("max_clusters must be an integer from 2 to 10.")
    values = scale_customer_features(features, **({"feature_weights": feature_weights} if feature_weights is not None else {}))
    upper = min(max_clusters, len(features) - 1, len(features.drop_duplicates()))
    if upper < 2:
        raise ValueError("Automatic selection requires at least three rows and two distinct profiles.")
    best_count, best_score = None, -float("inf")
    for count in range(2, upper + 1):
        labels = KMeans(n_clusters=count, random_state=random_state, n_init=10).fit_predict(values)
        score = segmentation_silhouette_score(values, labels, random_state=random_state)
        if score is not None and score > best_score:
            best_count, best_score = count, score
    if best_count is None:
        raise ValueError("No usable automatic segmentation was found.")
    return best_count


def segment_customers(csv_file, n_clusters=3, auto_clusters=False, max_display_rows=20, delimiter=",", random_state=42, id_column=None, include_distances=False, max_rows=None, auto_max_clusters=6, spend_weight=1.0, visits_weight=1.0, encoding="utf-8", max_bytes=None, expected_sha256=None):
    try:
        weights = [spend_weight, visits_weight]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 1e-12 <= v <= 1000 for v in weights):
            raise ValueError("Feature weights must be finite numbers from 1e-12 to 1000.")
        weight_options = {"feature_weights": weights} if weights != [1.0, 1.0] else {}
        if isinstance(auto_max_clusters, bool) or not isinstance(auto_max_clusters, int) or not 2 <= auto_max_clusters <= 10:
            raise ValueError("auto_max_clusters must be an integer from 2 to 10.")
        if not auto_clusters and auto_max_clusters != 6:
            raise ValueError("A custom automatic cluster budget requires auto_clusters.")
        if not isinstance(include_distances, bool):
            raise ValueError("include_distances must be a boolean.")
        validate_random_state(random_state)
        if not isinstance(max_display_rows, int) or isinstance(max_display_rows, bool) or max_display_rows < 1:
            raise ValueError("max_display_rows must be a positive integer.")
        if id_column is not None and (not isinstance(id_column, str) or not id_column.strip() or id_column in FEATURE_COLUMNS or id_column in {"segment", "segment_name"}):
            raise ValueError("Customer ID column must be a distinct source identifier.")
        df = _read_csv_input(csv_file, delimiter, **({"expected_sha256": expected_sha256} if expected_sha256 is not None else {}), **({"max_bytes": max_bytes} if max_bytes is not None else {}), **({"encoding": encoding} if encoding != "utf-8" else {}), **({"max_rows": max_rows} if max_rows is not None else {}), **({"converters": {id_column: lambda value: "__id__" + value}} if id_column is not None else {}))
        if id_column is not None:
            if id_column not in df.columns:
                raise ValueError("Customer ID column was not found.")
            # Some pandas versions recognize NA after applying converters.
            # A temporary prefix protects literal IDs; it never leaves parsing.
            df[id_column] = df[id_column].astype("string").str.slice(6)
            identifiers = df[id_column].str.strip()
            if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
                raise ValueError("Customer IDs must be non-empty and unique.")
        if include_distances and "distance_to_centroid" in df.columns:
            raise ValueError("distance_to_centroid is reserved for generated diagnostics.")
        missing = set(FEATURE_COLUMNS).difference(df.columns)
        if missing:
            raise ValueError(f"CSV must include: {', '.join(sorted(missing))}")
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer greater than or equal to 2.")

        if df[FEATURE_COLUMNS].apply(lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))).any().any():
            raise ValueError("Customer features must be numeric, not booleans.")
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
            n_clusters = select_cluster_count(features, random_state=random_state, **weight_options, **({"max_clusters": auto_max_clusters} if auto_max_clusters != 6 else {}))
        if len(df) < n_clusters:
            raise ValueError(f"At least {n_clusters} customers are required.")
        if len(features.drop_duplicates()) < n_clusters:
            raise ValueError(f"At least {n_clusters} distinct customer profiles are required.")

        scaled_features = scale_customer_features(features, **weight_options)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled_features)
        if include_distances:
            distances = np.linalg.norm(scaled_features - model.cluster_centers_[labels], axis=1)
            if not np.isfinite(distances).all():
                raise ValueError("Centroid distances must be finite.")
            df["distance_to_centroid"] = distances
        df["segment"] = labels
        df = assign_segment_names(df)
        quality_score = segmentation_silhouette_score(scaled_features, labels, random_state=random_state)
        df.attrs["silhouette_score"] = quality_score
        df.attrs["cluster_count"] = n_clusters
        if weight_options:
            df.attrs["feature_weights"] = dict(zip(FEATURE_COLUMNS, weights))

        columns = (["customer_id"] if "customer_id" in df.columns else []) + FEATURE_COLUMNS + ["segment", "segment_name"]
        print("\n===== Customer Segmentation Results =====")
        print(df[columns].head(max_display_rows).to_string(index=False))
        if len(df) > max_display_rows:
            print(f"Showing {max_display_rows} of {len(df)} customers; summaries and exports include every row.")
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



def export_segments_csv_atomic(result, output_path, escape_formulas=False):
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
        export = escape_csv_formulas(result) if escape_formulas else result
        export.to_csv(temporary_path, index=False)
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
    parser.add_argument("--max-display-rows", type=int, default=20, help="Maximum customer rows printed; exports remain complete")
    parser.add_argument("--escape-formulas", action="store_true", help="Prefix formula-like text with an apostrophe in CSV exports; calculations remain unchanged")
    parser.add_argument("--delimiter", choices=(",", ";", "tab", "|"), default=",", help="Input CSV delimiter; exports remain comma-separated")
    parser.add_argument("--random-state", type=int, default=42, help="Seed cluster fitting, automatic selection and silhouette sampling")
    parser.add_argument("--id-column", help="Require unique non-empty customer IDs and preserve their leading zeros")
    parser.add_argument("--include-distances", action="store_true", help="Include each customer’s Euclidean distance to its fitted centroid in scaled feature space")
    parser.add_argument("--max-input-rows", type=int, help="Reject CSV files exceeding this raw row budget (1..10000000)")
    parser.add_argument("--max-clusters", type=int, help="Automatic selection upper bound (2..10)")
    parser.add_argument("--spend-weight", type=float, default=1.0)
    parser.add_argument("--visits-weight", type=float, default=1.0)
    parser.add_argument("--encoding", choices=("utf-8", "utf-8-sig", "cp1252", "latin-1"), default="utf-8", help="Input CSV text encoding; output encoding remains UTF-8")
    parser.add_argument("--max-input-bytes", type=int, help="Snapshot and bound each decompressed CSV input (1..67108864 bytes)")
    parser.add_argument("--expect-input-sha256", help="Require the main CSV input to match this SHA-256 of decompressed bytes")
    args = parser.parse_args(argv)
    if args.max_input_bytes is not None and not 1 <= args.max_input_bytes <= 64 * 1024 * 1024:
        parser.error("--max-input-bytes must be from 1 to 67108864")
    if args.max_clusters is not None and (not args.auto_clusters or not 2 <= args.max_clusters <= 10):
        parser.error("--max-clusters requires --auto-clusters and a value from 2 to 10")
    if args.max_input_rows is not None and not 1 <= args.max_input_rows <= 10_000_000:
        parser.error("--max-input-rows must be from 1 to 10000000")
    delimiter = "\t" if args.delimiter == "tab" else args.delimiter
    try:
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
    except (OSError, RuntimeError) as exc:
        print(f"❌ Could not validate export paths: {exc}")
        return 1
    options = {"auto_clusters": True} if args.auto_clusters else {}
    if args.max_clusters is not None:
        options["auto_max_clusters"] = args.max_clusters
    if args.spend_weight != 1.0 or args.visits_weight != 1.0:
        options.update(spend_weight=args.spend_weight, visits_weight=args.visits_weight)
    options["max_display_rows"] = args.max_display_rows
    if args.include_distances:
        options["include_distances"] = True
    if args.id_column is not None:
        options["id_column"] = args.id_column
    if args.random_state != 42:
        options["random_state"] = args.random_state
    if args.encoding != "utf-8":
        options["encoding"] = args.encoding
    if args.max_input_bytes is not None:
        options["max_bytes"] = args.max_input_bytes
    if args.expect_input_sha256 is not None:
        options["expected_sha256"] = args.expect_input_sha256
    if args.max_input_rows is not None:
        options["max_rows"] = args.max_input_rows
    if delimiter != ",":
        options["delimiter"] = delimiter
    result = segment_customers(args.csv_file, n_clusters=args.clusters, **options)
    if result is None:
        return 1
    if output_path is not None:
        try:
            export_segments_csv_atomic(result, output_path, **({"escape_formulas": True} if args.escape_formulas else {}))
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
