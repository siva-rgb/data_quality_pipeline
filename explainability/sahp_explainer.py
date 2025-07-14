import shap
import pandas as pd

def explain_anomaly(model, data_sample, feature_names):
    """
    Use SHAP to explain a single row's anomaly based on reconstruction model.
    """
    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(pd.DataFrame(data_sample, columns=feature_names))
    return shap_values
