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
