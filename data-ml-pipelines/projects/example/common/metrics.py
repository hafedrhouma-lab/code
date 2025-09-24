import numpy as np

def recall_10(df, _builtin_metrics):
    """Calculate Recall@k based on the chain rank of the clicked item."""
    recall_at_k = (df['prediction'] <= 10).mean()
    return recall_at_k

def custom_accuracy(df, _builtin_metrics):
    return np.mean(df["target"] == df["prediction"])

