import pandas as pd
import numpy as np
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime

def train_tabnet_schema_model(df, categorical_cols, numerical_cols):
    """
    Train a TabNet model to learn the data structure.
    """
    df = df.copy()

    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Combine all columns
    all_cols = categorical_cols + numerical_cols
    X = df[all_cols].values
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    X_t = df[all_cols].values.astype(np.float32)
    X_train_uns, X_val_uns = train_test_split(X_t, test_size=0.2, random_state=42)
    print(X_train)
    print('X Value',X_val)
    # Pretrain (optional)
    unsupervised_model = TabNetPretrainer(optimizer_fn=torch.optim.Adam,
                                          optimizer_params=dict(lr=2e-2),
                                          mask_type='entmax')
    unsupervised_model.fit(
    X_train=X_train_uns,
    eval_set=[X_val_uns],
    pretraining_ratio=0.8,
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False)


    # Train TabNet to reconstruct inputs
    tabnet_model = TabNetRegressor(verbose=0)
    tabnet_model.fit(X_train=X_train, 
                     y_train=X_train, 
                     eval_set=[(X_val, X_val)],
                     from_unsupervised=unsupervised_model)

    return tabnet_model, encoders, scaler, all_cols

def infer_schema_violations(model, df, encoders, scaler, all_cols, threshold=0.1, mode="percentile"):
    """
    Use trained TabNet model to flag schema-inconsistent rows.

    Parameters:
    - model: trained TabNet model
    - df: input DataFrame
    - encoders: dictionary of fitted LabelEncoders
    - scaler: fitted scaler (e.g., StandardScaler)
    - all_cols: list of feature column names used in training
    - threshold: 
        * If mode="percentile", this is a float (e.g., 0.1 = top 10% error rows flagged)
        * If mode="absolute", this is a fixed MSE error threshold
    - mode: "percentile" or "absolute"
    """

    import numpy as np
    df_copy = df.copy()

    def safe_label_transform(le, values):
        known_classes = set(le.classes_)
        transformed = []
        for val in values:
            if val in known_classes:
                transformed.append(le.transform([val])[0])
            else:
                transformed.append(-1)  # fallback for unseen label
        return transformed

    # Encode categoricals safely
    for col, le in encoders.items():
        df_copy[col] = safe_label_transform(le, df_copy[col].astype(str))

    # Scale numeric columns
    df_copy[scaler.feature_names_in_] = scaler.transform(df_copy[scaler.feature_names_in_])
    X = df_copy[all_cols].values

    # Predict and calculate reconstruction error
    preds = model.predict(X)
    mse = np.mean((preds - X) ** 2, axis=1)

    # Determine outlier threshold
    if mode == "percentile":
        cutoff = np.percentile(mse, 100 - (threshold * 100))
    elif mode == "absolute":
        cutoff = threshold
    else:
        raise ValueError("mode must be 'percentile' or 'absolute'")

    outliers = np.where(mse > cutoff)[0]

    return [
        {
            "row": int(i),
            "reconstruction_error": float(mse[i])
        }
        for i in outliers
    ]

def explain_anomalies_per_column(model, df, encoders, scaler, all_cols, threshold=0.1, mode="percentile"):
    """
    Return row-wise reconstruction errors, and per-column contributions for flagged anomalies.

    Returns:
        List of dicts, each with:
        - row index
        - total reconstruction error
        - per-column error breakdown
    """
    import numpy as np
    df_copy = df.copy()

    def safe_label_transform(le, values):
        known_classes = set(le.classes_)
        transformed = []
        for val in values:
            if val in known_classes:
                transformed.append(le.transform([val])[0])
            else:
                transformed.append(-1)
        return transformed

    # Encode categoricals
    for col, le in encoders.items():
        df_copy[col] = safe_label_transform(le, df_copy[col].astype(str))

    # Scale numeric columns
    df_copy[scaler.feature_names_in_] = scaler.transform(df_copy[scaler.feature_names_in_])
    X = df_copy[all_cols].values

    # Predict and compute error per column
    preds = model.predict(X)
    errors = (preds - X) ** 2  # shape: (n_rows, n_cols)
    total_error = errors.mean(axis=1)

    # Determine which rows are anomalies
    if mode == "percentile":
        cutoff = np.percentile(total_error, 100 - (threshold * 100))
    elif mode == "absolute":
        cutoff = threshold
    else:
        raise ValueError("mode must be 'percentile' or 'absolute'")

    outlier_indices = np.where(total_error > cutoff)[0]
    # Calculate MSE for each row
    # This is the overall reconstruction error for each row
    mse = np.mean((model.predict(X) - X) ** 2, axis=1)
    save_error_distribution_plot(mse, threshold=0.1, mode="percentile")

    # Build output
    results = []
    for i in outlier_indices:
        row_result = {
            "row": int(i),
            "reconstruction_error": float(total_error[i]),
            "column_errors": {
                col: float(errors[i, j]) for j, col in enumerate(all_cols)
            }
        }
        results.append(row_result)

    return results


def save_error_distribution_plot(mse_values, threshold=0.1, mode="percentile", log_dir="logs", filename_prefix="error_dist"):
    """
    Plots and saves the reconstruction error distribution as a PNG.

    Parameters:
    - mse_values: array or list of reconstruction errors
    - threshold: float, either percentile or absolute value
    - mode: "percentile" or "absolute"
    - log_dir: directory to save plot in
    - filename_prefix: prefix for the output image file
    """
    os.makedirs(log_dir, exist_ok=True)

    # Determine cutoff line
    if mode == "percentile":
        cutoff = np.percentile(mse_values, 100 - (threshold * 100))
        threshold_label = f"{int(threshold * 100)}th Percentile"
    elif mode == "absolute":
        cutoff = threshold
        threshold_label = f"Error = {cutoff:.4f}"
    else:
        raise ValueError("mode must be 'percentile' or 'absolute'")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(mse_values, bins=50, color="skyblue", edgecolor="black")
    plt.axvline(cutoff, color="red", linestyle="--", label=f"Threshold ({threshold_label})")
    plt.xscale('log')
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Number of Rows")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.png")
    plt.savefig(filepath)
    plt.close()

    print(f"Reconstruction error plot saved to: {filepath}")
