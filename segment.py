import pandas as pd
from sklearn.cluster import KMeans
import sys

def segment_customers(csv_file):
    try:
        df = pd.read_csv(csv_file)

        if not {"total_spent", "visits"}.issubset(df.columns):
            print("❌ CSV must include 'total_spent' and 'visits' columns.")
            return

        X = df[["total_spent", "visits"]]

        kmeans = KMeans(n_clusters=3, random_state=42)
        df["segment"] = kmeans.fit_predict(X)

        print("\n===== Customer Segmentation Results =====")
        print(df[["customer_id", "total_spent", "visits", "segment"]])
        print("=========================================\n")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python segment.py <csv_file>")
        return

    segment_customers(sys.argv[1])

if __name__ == "__main__":
    main()
