import numpy as np
from scipy.stats import ks_2samp

def calculate_psi(expected, actual, buckets=10):
    def scale_range(data):
        return np.histogram(data, bins=buckets)[0] / len(data)

    expected_percents = scale_range(expected)
    actual_percents = scale_range(actual)
    
    psi_values = (expected_percents - actual_percents) * np.log(
        (expected_percents + 1e-5) / (actual_percents + 1e-5)
    )
    
    return np.sum(psi_values)

def detect_drift_ks(reference_df, new_df, alpha=0.05):
    drift_results = {}

    for col in reference_df.columns:
        ref_data = reference_df[col].dropna()
        new_data = new_df[col].dropna()

        if len(ref_data) > 0 and len(new_data) > 0:
            stat, p_value = ks_2samp(ref_data, new_data)
            drift_results[col] = {
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < alpha)
            }

    return drift_results
