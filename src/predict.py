from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def get_predict(model: BaseEstimator,
                X,
                return_proba:bool = False,
                threshold:float = 0.5):
    if return_proba:
        return model.predict_proba(X)[:, 1]
    
    if threshold == 0.5:
        return model.predict(X)
    
    proba = model.predict_proba(X)[:, 1]
    return (proba >=threshold).astype(int)


def get_metrics(y_true, y_pred, y_proba=None):
    metrics = {
          "accuracy": accuracy_score(y_true, y_pred),
          "precision": precision_score(y_true, y_pred),
          "recall": recall_score(y_true, y_pred),
          "f1": f1_score(y_true, y_pred)
          }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics
