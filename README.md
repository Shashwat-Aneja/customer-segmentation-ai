# Customer Segmentation AI

A machine learning project that performs customer segmentation using K-Means clustering.  
This tool separates customers based on their transaction patterns or engagement metrics, helping businesses make data-driven decisions.

---

## 📊 Features
- Groups customers into segments (default: 3 clusters)
- Reads data from CSV file
- Identifies high-value / low-value / moderate customers
- Outputs segmentation results with cluster labels

---

## 🧪 Example CSV Format
```
customer_id,total_spent,visits
1,15000,10
2,5000,5
3,22000,14
4,1800,3
5,9500,7
```

---

## 🚀 Usage

### 1. Clone repository
```bash
git clone https://github.com/Shashwat-Aneja/customer-segmentation-ai
cd customer-segmentation-ai
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn
```

### 3. Run
```bash
python segment.py data.csv
```

---

## 📁 Structure
```
customer-segmentation-ai/
│
├── segment.py
└── README.md
```

---

## 📌 Future Enhancements
- Visualize clusters using matplotlib
- Add more features (average purchase value, last purchase date)
- Export segmentation results as JSON or CSV
- Integrate with AI Business Assistant

---

Built by **Shashwat Aneja**
