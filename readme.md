# 🧠 Data Quality Solution using Statistical, Machine Learning, and Deep Learning Approaches

## 📌 Project Overview

This project provides a comprehensive **Data Quality (DQ) Solution** that combines **statistical analysis**, **machine learning (ML)**, and **deep learning (DL)** to automatically generate data quality reports and suggest mitigation strategies. It addresses multiple dimensions of data quality and helps organizations improve trust and usability of their datasets.

---

## 🎯 Objectives

- Capture and assess data quality across critical dimensions.
- Detect and visualize data quality issues.
- Apply ML and DL-based techniques to improve and impute faulty or missing data.
- Generate interpretable, actionable reports for users.

---

## 📊 Data Quality Dimensions Covered

The project identifies and mitigates issues across the following **data quality dimensions**:

- **Completeness**
- **Consistency**
- **Accuracy**
- **Validity**
- **Timeliness**

---

## 🧩 Dimension-wise Breakdown

### ✅ Completeness

- 🔍 Detect missing values in each column.
- 📊 Calculate % of missing data per column.
  - Drop values if missingness is <5%.
- 🧠 Identify type of missingness:
  - MCAR (Missing Completely at Random)
  - MAR (Missing at Random)
  - MNAR (Missing Not at Random)
- 🛠️ Impute missing data for categorical and numerical columns.
- 🧪 Techniques for imputation:
  - **KNN Imputer**
  - **MICE (Multivariate Imputation by Chained Equations)**
  - **Autoencoders**
- 📈 Evaluate the performance of each imputation technique.

---

### 🔁 Consistency

- 🕵️ Detect anomalies in the dataset.
- 🧪 Check if numeric columns contain string values (and vice versa).
- 💡 Use **SHAP** to explain anomaly-contributing columns.
- 🔡 Detect inconsistent text patterns in categorical columns.

---

### 🎯 Accuracy

- 🔄 Use **autoencoder reconstruction error** to assess accuracy.
- 📏 Check whether numerical column values fall within expected ranges.

---

### 🔐 Validity

- ✉️ Perform pattern-based checks for:
  - **Email validity**
  - **Phone number validity**
- ✔️ Validate against standard regex formats.

---

### ⏱️ Timeliness

- ⌛ **Latency** = Time Data is Needed − Time Data is Available
- 📅 **Age of Data** = Current Time − Last Update Time
- 🟢 **Data Freshness Score** (0 to 1 scale):
  - Requires user to specify an "insertion timestamp" column.
- 🔄 Perform **schema drift** and **data drift** detection.
- 🔁 Build **autoencoders for data reconciliation** over time.

---

## 🧰 Technologies Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**, **Keras**, **PyTorch**
- **SHAP** for explainability
- **Seaborn / Matplotlib** for visualization
- **Regex** for pattern-based checks
- **Apache Airflow / Streamlit** (optional for orchestration / UI)

---

## 📈 Future Enhancements

- Integrate with **BigQuery**, **Snowflake**, or **AWS Redshift** for large-scale deployment.
- Add a **Streamlit-based dashboard** for interactive report viewing.
- Enable **user-defined validation rules**.
- Include **drift monitoring over time** with alerts.

---

## 📂 Folder Structure (Sample)

