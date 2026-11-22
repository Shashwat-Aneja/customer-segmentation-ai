# Customer Segmentation AI

A K-Means clustering tool that groups customers using spending and visit frequency.

## Features
- Reads customer records from CSV
- Standardizes spending and visit features before clustering so currency scale does not dominate
- Uses deterministic K-Means initialization for reproducible runs
- Rejects missing, non-numeric, non-finite, or insufficient customer profiles

## Input CSV
Required columns: `total_spent` and `visits`. An optional `customer_id` column is included in output.

```csv
customer_id,total_spent,visits
1,15000,10
2,5000,5
3,22000,14
4,1800,3
5,9500,7
```

At least three distinct customer profiles are needed for the default three clusters.

## Run
```bash
pip install pandas scikit-learn
python segment.py data.csv
```

Cluster numbers are identifiers, not ranked customer-value labels. Inspect each cluster's feature profile before assigning business meaning.

Developed by **Shashwat Aneja**.

`--summary-output segments.json` saves segment sizes, average behavior and silhouette score in an atomic JSON report. The summary cannot overwrite either the source CSV or customer-level export.

Use `--auto-clusters` instead of `--clusters` to evaluate 2–6 feasible cluster counts on standardized features and choose the highest silhouette score. Selection is deterministic, bounded, and requires at least three rows and two distinct profiles.

Silhouette evaluation deterministically samples at most 2,000 customers while including each cluster. Large customer files no longer require all-pairs scoring for every candidate cluster count; quality values for large files are sample estimates.

Console customer previews default to 20 rows, configurable with `--max-display-rows`. Model fitting, profiles and CSV exports still include all customers.

Use `--escape-formulas` when CSV exports will be opened in spreadsheet software. Formula-like text cells beginning with `=`, `+`, `-` or `@` after leading whitespace receive an apostrophe; numeric cells and in-memory calculations remain unchanged. The default export preserves original text.

`--delimiter ";"`, `--delimiter tab` and `--delimiter "|"` support alternate input CSV separators, including quoted fields. Duplicate-header validation uses the selected separator. The default input and all CSV exports remain comma-separated.

`--random-state N` controls K-Means fitting, automatic cluster selection and sampled silhouette scoring consistently. The default remains 42; repeatability assumes the same input, library versions and environment.

Segment summaries include `customer_share_percent`, with all assigned customers forming the denominator. Missing segment identifiers are rejected rather than silently omitted from coverage.
