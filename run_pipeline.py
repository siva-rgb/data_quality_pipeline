import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoders import SchemaAutoencoder
from models.drift_detector import detect_drift_ks, calculate_psi
from text.nlp_consistency import detect_text_inconsistencies_columnwise


def detect_type_inconsistencies(df, threshold=0.9):
    """
    Flags rows where a column is mostly numeric, but contains non-numeric entries.
    Returns a list of dicts with row index, column name, and the invalid value.
    """
    issues = []

    for col in df.columns:
        # Skip columns already known as categorical
        if df[col].dtype == 'object':
            continue

        numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
        ratio = numeric_count / len(df)

        # If mostly numeric
        if ratio >= threshold:
            for idx, val in df[col].items():
                try:
                    float(val)
                except (ValueError, TypeError):
                    issues.append({
                        "row": int(idx),
                        "column": col,
                        "invalid_value": str(val)
                    })

    return issues

def explain_anomaly_per_column(input_row, reconstructed_row, feature_names):
    """
    Returns a dictionary of reconstruction error per feature (MSE).
    """
    errors = (input_row - reconstructed_row) ** 2
    return {feature_names[i]: float(errors[i]) for i in range(len(feature_names))}


def run_pipeline():
    os.makedirs("output", exist_ok=True)

    # Sample dataset (replace this with your own data loading)
    # np.random.seed(42)
    # df = pd.DataFrame({
    #     'age': np.random.normal(30, 5, 100),
    #     "gender": np.random.choice(["M", "F", "f", "Other", "m","female"],100),
    #     'salary': np.random.normal(50000, 10000, 100),
    #     'city': np.random.choice(['New York', 'NYC', 'new york', 'Los Angeles', 'LA'], 100)
    # })

    df= pd.read_csv("data/raw/synthetic_data_for_data_quality.csv")
     # Drop id column form dataset
    if 'Item_Iden' in df.columns:
        df.drop(columns=['Item_Iden'], inplace=True)

    # ---------- Schema Detection ----------
    # numerical column 
    # feature_names= ['age', 'salary']
    feature_names = ['Item_Weig', 'Item_Visib', 'Item_MRP', 'Item_Sales', 'Outlet_Est']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_names])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = SchemaAutoencoder(input_dim=X_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(X_tensor), batch_size=16, shuffle=True)
    model.train()
    for epoch in range(20):
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch[0])
            loss = criterion(output, batch[0])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, axis=1).numpy()

    threshold = np.percentile(errors, 95)
    schema_issues = []

    for i, error in enumerate(errors):
        if error > threshold:
            per_col_error = explain_anomaly_per_column(
                input_row=X_tensor[i].numpy(),
                reconstructed_row=reconstructed[i].numpy(),
                feature_names=feature_names
            )
            schema_issues.append({
                "row": int(i),
                "reconstruction_error": float(error),
                "column_errors": per_col_error
            })

    # ---------- Text Inconsistency Check ----------
    #categorial columns
    # categorical_columns= ['city','gender']
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    text_anomalies = detect_text_inconsistencies_columnwise(df,categorical_columns)

    print(text_anomalies)
    # ---------- Drift Detection (Simulated Example) ----------
    drift_results = detect_drift_ks(df[feature_names], df[feature_names])  # Simulate same data

        # ---------- Type Consistency Check ----------
    type_inconsistencies = detect_type_inconsistencies(df)

    # -------------- Schema Issues Summary --------------#

    # ---------- Output JSON ----------
    report = {
        "schema_issues": schema_issues,
        "text_issues": {
            "column": "city",
            "anomalies": text_anomalies
        },
        "drift": drift_results,
        "type_inconsistencies": type_inconsistencies
    }

    with open("output/data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Data quality report saved to output/data_quality_report.json")


if __name__ == "__main__":
    run_pipeline()