import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction


def binary_classification_metrics(predictions, labels, threshold=0.5):
    # Apply softmax on predictions
    probs = F.softmax(torch.Tensor(predictions), dim=1)

    # Use argmax to convert to binary predictions
    y_pred = torch.argmax(probs, dim=1)

    # Compute metrics
    y_true = labels
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    # Return as dictionary
    metrics = {'accuracy': accuracy, 'f1': f1}
    return metrics


def compute_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = binary_classification_metrics(predictions=logits, labels=p.label_ids)
    return result
