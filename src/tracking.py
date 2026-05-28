import mlflow
import mlflow.sklearn
from pathlib import Path

EXPERIMENT_NAME = "mental_health_depression"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH}"

def setup_mlflow(experiment_name: str = EXPERIMENT_NAME):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

def log_experiment(
        run_name: str,
        model,
        params: dict,
        metrics: dict,
        tags: dict|None = None
):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)
        
        mlflow.sklearn.log_model(model, name="model")

