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

Seekable caller-owned text and UTF-8 binary CSV streams now receive the same duplicate-header checks as file paths, before pandas can rename ambiguous columns. Validation restores the starting position and leaves caller streams open; nonseekable stream objects are rejected explicitly.

`--id-column customer_id` optionally requires non-empty unique customer identities before clustering. Identifier text preserves leading zeros; whitespace is ignored for uniqueness checks. Duplicate identities fail explicitly instead of being dropped or silently giving a customer extra weight.

CSV paths ending in `.gz`, `.bz2`, or `.xz` are supported without unpacking a separate file. Header validation reads the same compression format as pandas and still rejects duplicate columns, including BOM-prefixed quoted headers and alternate delimiters. Archive containers such as ZIP/TAR are outside this feature.

`--include-distances` adds a finite Euclidean `distance_to_centroid` for each customer in the model’s scaled feature space. Distances help inspect within-segment spread; they are not probabilities or an automatic outlier classification. Existing source columns with that name are rejected before generating it.

Corrupt compressed bodies, invalid DEFLATE blocks and header-parser limit failures use the normal validation-error path rather than escaping as EOFError, zlib.error or csv.Error. Header inspection restores caller stream position/ownership on these failures. No invalid input is exported.

### Optional raw CSV row budget

Use `--max-input-rows 100000` to reject an input file with more than that many data rows before modeling or export. The budget is per file, including comparison/training/scoring inputs, and applies before filtering or aggregation. The default remains unlimited. Values must be integers from 1 to 10,000,000. Only the budget plus one rows are loaded; oversized inputs fail rather than being silently truncated. This is a row limit, not a byte or field-size limit. Python callers can use `max_rows=`.
