from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
import joblib


def train(Xtrain,
        ytrain,
        pipeline: Pipeline,
        params_grid: dict):
    

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(pipeline,
                      param_grid=params_grid,
                      n_jobs=-1,
                      cv=cv,
                      scoring="roc_auc")
    
    gs.fit(Xtrain, ytrain)
    return gs


def model_dump(model: BaseEstimator, path):
    try:
        joblib.dump(model, path)
    except Exception as e:
        raise Exception(f"Не удалось загрузить модель {e}")
    
def model_load(path):
    return joblib.load(path)