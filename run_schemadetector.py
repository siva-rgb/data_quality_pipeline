import pandas as pd
from models.schema_detection import train_tabnet_schema_model, infer_schema_violations, explain_anomalies_per_column
import numpy as np
import random
import logging
import os
import json
from datetime import datetime
# Set random seed for reproducibility

def save_log(data, log_dir="logs", filename_prefix="anomaly_log"):
    """
    Save data (list or dict) as a JSON log file inside a logs folder.

    Parameters:
    - data: list or dict to save
    - log_dir: name of the directory to save logs in (default: 'logs')
    - filename_prefix: prefix for the log file name
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.json")

    # Save the data
    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Log saved to: {log_path}")


# np.random.seed(42)
# df = pd.DataFrame({
#     "age": np.random.choice([25, 30, 45, "unknown", 35],100),
#     "gender": np.random.choice(["M", "F", "F", "Other", "F","female"],100),
#     "income": np.random.choice([40000,45000, 61000, 50000, "error", 60000, 70000],100)
# })
# df.to_csv("sample_data.csv", index=False)
# Fix potential type issues for the test
# df["age"] = pd.to_numeric(df["age"], errors="coerce")
# df["income"] = pd.to_numeric(df["income"], errors="coerce")
df= pd.read_csv("data/raw/synthetic_data_for_data_quality.csv")
     # Drop id column form dataset

df.drop(columns=['Item_Iden','Outlet_Ide'], inplace=True)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = ['Item_Weig', 'Item_Visib', 'Item_MRP', 'Item_Sales', 'Outlet_Est']

model, enc, scaler, cols = train_tabnet_schema_model(df.dropna(), cat_cols, num_cols)

violations = infer_schema_violations(model, df.fillna(0), enc, scaler, cols)
explained_violations = explain_anomalies_per_column(model, df.fillna(0), enc, scaler, cols)
save_log(violations)

save_log(explained_violations, filename_prefix="anomaly_explanation")
#save the logs in a log file in logs folder