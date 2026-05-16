
# 🛒 E-Commerce Orders — Data Analysis & Machine Learning

A complete end-to-end data science project on an e-commerce orders dataset:  
**exploratory data analysis → feature engineering → 3 machine learning models → business insights**.



## 📁 Project Structure

```
├── Dataset_for_Data_Analytics.xlsx   # Original raw dataset
├── complete_analysis_ml.py           # Full analysis + ML code (run this)
├── decode_label.py                   # Label encoder/decoder utility
├── submission.csv                    # Final predictions output
├── charts/
│   ├── 01_univariate_analysis.png
│   ├── 02_bivariate_analysis.png
│   ├── 03_time_series.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_regression_feature_importance.png
│   ├── 06_regression_actual_vs_predicted.png
│   ├── 07_classification_confusion_matrix.png
│   ├── 08_classification_feature_importance.png
│   ├── 09_clustering_elbow.png
│   ├── 10_clustering_segments.png
│   └── 11_segment_sizes.png
└── README.md




## 📊 Dataset Overview

| Property | Value |
|---|---|
| Rows | 1,200 |
| Columns | 14 |
| Date Range | Jan 2023 – Jun 2025 |
| Missing Values | 309 (CouponCode only) |

### Columns

| Column | Type | Description |
|---|---|---|
| `OrderID` | str | Unique order identifier |
| `Date` | datetime | Order date |
| `CustomerID` | str | Customer identifier |
| `Product` | categorical | 7 products (Laptop, Phone, Tablet, Monitor, Chair, Desk, Printer) |
| `Quantity` | int | Units ordered (1–5) |
| `UnitPrice` | float | Price per unit ($11 – $700) |
| `ShippingAddress` | str | Delivery address |
| `PaymentMethod` | categorical | Cash, Credit Card, Debit Card, Gift Card, Online |
| `OrderStatus` | categorical | Cancelled, Delivered, Pending, Returned, Shipped |
| `TrackingNumber` | str | Unique tracking ID |
| `ItemsInCart` | int | Items browsed before purchase (1–10) |
| `CouponCode` | categorical | SAVE10 / FREESHIP / WINTER15 / (none) |
| `ReferralSource` | categorical | Email, Facebook, Google, Instagram, Referral |
| `TotalPrice` | float | Final order value |



## ⚙️ Installation

bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ecommerce-ml-analysis.git
cd ecommerce-ml-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl




## 🚀 How to Run

bash
# Run the full analysis (generates all charts + prints results)
python complete_analysis_ml.py

# Test the label decoder
python decode_label.py




## 🤖 Machine Learning Models

### Model 1 — Regression: Predict `TotalPrice`

**Goal:** Predict how much an order will cost.

| Metric | Score |
|---|---|
| R² | **0.9997** |
| MAE | $8.54 |
| RMSE | $14.37 |

**Top features:** `UnitPrice` and `Quantity` dominate. Product type and coupon also matter.



### Model 2 — Classification: Predict Delivery

**Goal:** Predict if an order will be `Delivered` (vs Cancelled/Returned/Shipped/Pending).

| Metric | Score |
|---|---|
| CV Accuracy | **79.8%** |
| Precision (Delivered) | 0.00* |
| Recall (Not Delivered) | 0.99 |

> ⚠️ Class imbalance: only 19.3% of orders are "Delivered". Consider SMOTE or class weighting for improvement.



### Model 3 — Clustering: Customer Segmentation

**Goal:** Group customers by purchasing behavior using K-Means (k=3).

| Segment | Customers | Avg Total Spend | Avg Order Value |
|---|---|---|---|
| 🏆 High-Value (VIP) | 435 | $1,908 | $1,908 |
| 🔁 Mid-Value (Regular) | 11 | $1,776 | $888 |
| 💤 Low-Value (Occasional) | 743 | $559 | $559 |



## 🔤 Using `decode_label.py`

After running the models, use the decoder to convert integer codes back to labels:

```python
from decode_label import decode_label, decode_dataframe
import pandas as pd

# Decode a single value
decode_label('Product_enc', 2)          # → 'Laptop'
decode_label('OrderStatus_enc', 0)      # → 'Cancelled'
decode_label('IsDelivered', 1)          # → 'Delivered'

# Decode an entire DataFrame
df = pd.read_csv('submission.csv')
df_decoded = decode_dataframe(df)
print(df_decoded.head())
```

### Encoding Reference

| Code | Product | PaymentMethod | OrderStatus | CouponCode | ReferralSource |
|---|---|---|---|---|---|
| 0 | Chair | Cash | Cancelled | FREESHIP | Email |
| 1 | Desk | Credit Card | Delivered | NoCoupon | Facebook |
| 2 | Laptop | Debit Card | Pending | SAVE10 | Google |
| 3 | Monitor | Gift Card | Returned | WINTER15 | Instagram |
| 4 | Phone | Online | Shipped | — | Referral |
| 5 | Printer | — | — | — | — |
| 6 | Tablet | — | — | — | — |



## 📄 `submission.csv` Columns

| Column | Description |
|---|---|
| `OrderID` | Original order ID |
| `CustomerID` | Original customer ID |
| `Date` | Order date |
| `Product` | Product name |
| `Quantity` | Units ordered |
| `UnitPrice` | Price per unit |
| `TotalPrice` | Actual total price |
| `Predicted_TotalPrice` | Model 1 prediction |
| `OrderStatus` | Actual order status |
| `Predicted_DeliveryLabel` | Model 2 prediction (Delivered / Not Delivered) |
| `CustomerSegment` | Model 3 cluster label |



## 💡 Key Business Insights

| Insight | Finding |
|---|---|
| 🏅 Top Revenue Product | **Chair** ($195,620 total) |
| 💳 Most Used Payment | **Online** |
| 📣 Best Referral Source | **Facebook** (avg $1,098/order) |
| ❌ Cancellation Rate | **20.8%** — needs investigation |
| 🎟️ Coupon Usage | **74.2%** of orders — high discount dependency |
| 👑 VIP Customers | **435 customers** drive $1,908 avg spend |



## 📈 Charts Preview



 ![Univariate](charts/01_univariate_analysis.png) | ![Bivariate](charts/02_bivariate_analysis.png) |
 ![TimeSeries](charts/03_time_series.png) | ![Heatmap](charts/04_correlation_heatmap.png) |



## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4c72b0)


## 👤 Author

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)



## 📜 License

# 🛒 E-Commerce Orders — Data Analysis & Machine Learning

A complete end-to-end data science project on an e-commerce orders dataset:  
**exploratory data analysis → feature engineering → 3 machine learning models → business insights**.



## 📁 Project Structure


├── Dataset_for_Data_Analytics.xlsx   # Original raw dataset
├── complete_analysis_ml.py           # Full analysis + ML code (run this)
├── decode_label.py                   # Label encoder/decoder utility
├── submission.csv                    # Final predictions output
├── charts/
│   ├── 01_univariate_analysis.png
│   ├── 02_bivariate_analysis.png
│   ├── 03_time_series.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_regression_feature_importance.png
│   ├── 06_regression_actual_vs_predicted.png
│   ├── 07_classification_confusion_matrix.png
│   ├── 08_classification_feature_importance.png
│   ├── 09_clustering_elbow.png
│   ├── 10_clustering_segments.png
│   └── 11_segment_sizes.png
└── README.md



## 📊 Dataset Overview

| Property | Value |
|---|---|
| Rows | 1,200 |
| Columns | 14 |
| Date Range | Jan 2023 – Jun 2025 |
| Missing Values | 309 (CouponCode only) |

### Columns

| Column | Type | Description |
|---|---|---|
| `OrderID` | str | Unique order identifier |
| `Date` | datetime | Order date |
| `CustomerID` | str | Customer identifier |
| `Product` | categorical | 7 products (Laptop, Phone, Tablet, Monitor, Chair, Desk, Printer) |
| `Quantity` | int | Units ordered (1–5) |
| `UnitPrice` | float | Price per unit ($11 – $700) |
| `ShippingAddress` | str | Delivery address |
| `PaymentMethod` | categorical | Cash, Credit Card, Debit Card, Gift Card, Online |
| `OrderStatus` | categorical | Cancelled, Delivered, Pending, Returned, Shipped |
| `TrackingNumber` | str | Unique tracking ID |
| `ItemsInCart` | int | Items browsed before purchase (1–10) |
| `CouponCode` | categorical | SAVE10 / FREESHIP / WINTER15 / (none) |
| `ReferralSource` | categorical | Email, Facebook, Google, Instagram, Referral |
| `TotalPrice` | float | Final order value |



## ⚙️ Installation

bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ecommerce-ml-analysis.git
cd ecommerce-ml-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl




## 🚀 How to Run

bash
# Run the full analysis (generates all charts + prints results)
python complete_analysis_ml.py

# Test the label decoder
python decode_label.py




## 🤖 Machine Learning Models

### Model 1 — Regression: Predict `TotalPrice`

**Goal:** Predict how much an order will cost.

| Metric | Score |

| R² | **0.9997** |
| MAE | $8.54 |
| RMSE | $14.37 |

**Top features:** `UnitPrice` and `Quantity` dominate. Product type and coupon also matter.



### Model 2 — Classification: Predict Delivery

**Goal:** Predict if an order will be `Delivered` (vs Cancelled/Returned/Shipped/Pending).

| Metric | Score |
|---|---|
| CV Accuracy | **79.8%** |
| Precision (Delivered) | 0.00* |
| Recall (Not Delivered) | 0.99 |

> ⚠️ Class imbalance: only 19.3% of orders are "Delivered". Consider SMOTE or class weighting for improvement.

---

### Model 3 — Clustering: Customer Segmentation

**Goal:** Group customers by purchasing behavior using K-Means (k=3).

| Segment | Customers | Avg Total Spend | Avg Order Value |
|---|---|---|---|
| 🏆 High-Value (VIP) | 435 | $1,908 | $1,908 |
| 🔁 Mid-Value (Regular) | 11 | $1,776 | $888 |
| 💤 Low-Value (Occasional) | 743 | $559 | $559 |

---

## 🔤 Using `decode_label.py`

After running the models, use the decoder to convert integer codes back to labels:

```python
from decode_label import decode_label, decode_dataframe
import pandas as pd

# Decode a single value
decode_label('Product_enc', 2)          # → 'Laptop'
decode_label('OrderStatus_enc', 0)      # → 'Cancelled'
decode_label('IsDelivered', 1)          # → 'Delivered'

# Decode an entire DataFrame
df = pd.read_csv('submission.csv')
df_decoded = decode_dataframe(df)
print(df_decoded.head())
```

### Encoding Reference

| Code | Product | PaymentMethod | OrderStatus | CouponCode | ReferralSource |
|---|---|---|---|---|---|
| 0 | Chair | Cash | Cancelled | FREESHIP | Email |
| 1 | Desk | Credit Card | Delivered | NoCoupon | Facebook |
| 2 | Laptop | Debit Card | Pending | SAVE10 | Google |
| 3 | Monitor | Gift Card | Returned | WINTER15 | Instagram |
| 4 | Phone | Online | Shipped | — | Referral |
| 5 | Printer | — | — | — | — |
| 6 | Tablet | — | — | — | — |

---

## 📄 `submission.csv` Columns

| Column | Description |
|---|---|
| `OrderID` | Original order ID |
| `CustomerID` | Original customer ID |
| `Date` | Order date |
| `Product` | Product name |
| `Quantity` | Units ordered |
| `UnitPrice` | Price per unit |
| `TotalPrice` | Actual total price |
| `Predicted_TotalPrice` | Model 1 prediction |
| `OrderStatus` | Actual order status |
| `Predicted_DeliveryLabel` | Model 2 prediction (Delivered / Not Delivered) |
| `CustomerSegment` | Model 3 cluster label |

---

## 💡 Key Business Insights

| Insight | Finding |
|---|---|
| 🏅 Top Revenue Product | **Chair** ($195,620 total) |
| 💳 Most Used Payment | **Online** |
| 📣 Best Referral Source | **Facebook** (avg $1,098/order) |
| ❌ Cancellation Rate | **20.8%** — needs investigation |
| 🎟️ Coupon Usage | **74.2%** of orders — high discount dependency |
| 👑 VIP Customers | **435 customers** drive $1,908 avg spend |

---

## 📈 Charts Preview

| | |
|---|---|
| ![Univariate](charts/01_univariate_analysis.png) | ![Bivariate](charts/02_bivariate_analysis.png) |
| ![TimeSeries](charts/03_time_series.png) | ![Heatmap](charts/04_correlation_heatmap.png) |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4c72b0)

---

## 👤 Author

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📜 License


This project is open source under the [MIT License](LICENSE).