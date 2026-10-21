Customer segmentation guide

The CSV requires numeric `total_spent` and `visits`; `customer_id` is optional. The default is three clusters and the dataset must contain at least three distinct feature profiles.

Run `python segment.py sample_customers.csv` after installing `requirements.txt`. Features are standardized before K-means so different units do not dominate distances. Cluster numbers are arbitrary labels, not rankings. Inspect cluster centers in original units before assigning business meaning, and revalidate when the customer population changes.