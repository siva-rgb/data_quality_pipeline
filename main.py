# to get fata quality of the data statistical data quality
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import torch
import torch.nn as nn
import torch.optim as optim
from text.nlp_column_inconsistancy import detect_column_inconsistencies, detect_text_anomalies_dbscan 
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.preprocessing import StandardScaler
from models.autoencoders import SchemaAutoencoder
from models.drift_detector import detect_drift_ks, calculate_psi
from text.nlp_consistency import detect_text_inconsistencies_columnwise
from models.acc_autoencoder import Autoencoder
from utils.preprocess import preprocess_data
import os
import json
# import msno

quality_report_items= {}
def calculate_data_quality(df):
    # percentage of data mising for each column 
    columns= df.columns
    missing_col_pct={}
    for col in columns:
        missing_count = df[col].isnull().sum()
        total_count = len(df[col])
        missing_col_pct[col] = (missing_count / total_count) * 100 if total_count > 0 else 0
    # percentage of null in total data
    total_missing_cells = df.isnull().sum().sum()
    total_cells = df.size
    total_missing = (total_missing_cells / total_cells) * 100 if total_cells > 0 else 0
    msno.matrix(df)
    msno.heatmap(df)
    data_quality = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "data_types": df.dtypes.to_dict(),
        "quality_report_items": missing_col_pct,
        "total_missing_percentage": total_missing
    }
    return data_quality

# consitency check for the data
def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def check_data_consistency(df):
    consistency_report = {}
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    consistency_report['duplicate_rows'] = duplicate_rows

    # Check for unique values in categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    unique_values = {col: df[col].unique().tolist() for col in categorical_columns}
    consistency_report['unique_values'] = unique_values

    # Check for outliers in numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = {}
    for col in numerical_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()

    consistency_report['col_outliers'] = outliers
    #check if mejority column is numeric or categorical
    column_consistency = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            col= df[col].dropna().astype(str)
            num_count = col.apply(is_numeric).sum()
            cat_count= len(col) - num_count
            # Check for numeric consistency
            if num_count > cat_count:
                column_consistency[col.name] = f'{col}: mejority numeric {num_count} vs categorical {cat_count}'

            else:
                column_consistency[col.name] = f'{col}: mejority catagptical {cat_count} vs numerical {num_count}'

    consistency_report['column_consistency'] = column_consistency
        
    # get test inconsistency in categorical columns
    categorical_columns = ['Item_Fat_', 'Lo_Outlet_']
    result, report = detect_column_inconsistencies(df, categorical_columns)
    consistency_report['text_inconsistencies'] = result
    consistency_report['text_inconsistency_report'] = report
    # get text anomalies using dbscan
    col_anomalies_db= detect_text_anomalies_dbscan(df, categorical_columns)
    consistency_report['text_anomalies_dbscan'] = col_anomalies_db
    return consistency_report

def train_autoencoder(model, X_train, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        output = model(X_train)
        loss = loss_fn(output, X_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ===== Evaluate & Get Reconstruction Errors =====
def get_reconstruction_errors(model, X):
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
    return errors

# ===== Plot Errors =====
def plot_errors(errors, threshold=None):
    plt.figure(figsize=(10, 4))
    plt.plot(errors, label="Reconstruction Error")
    if threshold:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Autoencoder-based Data Accuracy Check")
    plt.legend()
    plt.grid(True)
    plt.show()

def data_accuracy_check(df):
    """
    Check the accuracy of the data by comparing it with a reference dataset.
    This is a placeholder function and should be implemented based on specific requirements.
    """
    accuracy_report = {}
    X, index, scaler = preprocess_data(df)
    model = Autoencoder(input_dim=X.shape[1])
    train_autoencoder(model, X)

    errors = get_reconstruction_errors(model, X)
    threshold = np.percentile(errors, 95)  # 95th percentile as threshold

    # Flag rows with high error
    inaccurate_rows = index[errors > threshold]
    print(f"Inaccurate Row Indices:\n{inaccurate_rows}")
    inacc_data= pd.DataFrame({'Index': index, 'Reconstruction_Error': errors, 'Inaccurate': errors > threshold})
    inacc_data_val= inacc_data[inacc_data['Inaccurate'] == True]
    plot_errors(errors, threshold)
    accuracy_report['inaccurate_rows'] = inacc_data_val.to_dict(orient='records')
    return accuracy_report

df= pd.read_csv("data/raw/synthetic_data_for_data_quality.csv")
completenes_reslut= calculate_data_quality(df)
quality_report_items['completeness'] = completenes_reslut
consistency_result = check_data_consistency(df)
quality_report_items['consistency'] = consistency_result
accuracy_result = data_accuracy_check(df)
quality_report_items['accuracy'] = accuracy_result
print("Data Quality Report:")
print(type(quality_report_items))

# save the data in json file in output folder
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, np.dtype):  # NEW LINE to handle np.dtypes
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json_str = json.dumps(quality_report_items, default=convert_numpy)
print(json_str)
with open("output/data_quality_dump.json", "w") as f:
        json.dump(json_str, f, indent=2)

print("✅ Data quality report saved to output/data_quality_dump.json")