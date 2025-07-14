import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# ===== Load and Preprocess Data =====
def preprocess_data(df):
    df = df.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return torch.tensor(X_scaled, dtype=torch.float32), df.index, scaler