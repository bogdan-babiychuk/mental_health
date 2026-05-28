from unittest.mock import MagicMock, patch

import pytest

from src import tracking
from src.tracking import log_experiment, setup_mlflow


@patch("src.tracking.mlflow")
def test_setup_mlflow_uses_sqlite_uri(mock_mlflow):
    setup_mlflow("my_experiment")

    mock_mlflow.set_tracking_uri.assert_called_once()
    uri = mock_mlflow.set_tracking_uri.call_args.args[0]
    assert uri.startswith("sqlite:///")
    assert "mlflow.db" in uri
    mock_mlflow.set_experiment.assert_called_once_with("my_experiment")


@patch("src.tracking.mlflow")
def test_setup_mlflow_uses_default_experiment(mock_mlflow):
    setup_mlflow()
    mock_mlflow.set_experiment.assert_called_once_with(tracking.EXPERIMENT_NAME)


@patch("src.tracking.mlflow")
def test_log_experiment_logs_params_metrics_and_model(mock_mlflow):
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
    model = MagicMock(name="model")

    log_experiment(
        run_name="run-1",
        model=model,
        params={"C": 1.0},
        metrics={"accuracy": 0.9},
    )

    mock_mlflow.start_run.assert_called_once_with(run_name="run-1")
    mock_mlflow.log_params.assert_called_once_with({"C": 1.0})
    mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.9})
    mock_mlflow.sklearn.log_model.assert_called_once_with(model, name="model")
    mock_mlflow.set_tags.assert_not_called()


@patch("src.tracking.mlflow")
def test_log_experiment_sets_tags_when_provided(mock_mlflow):
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

    log_experiment(
        run_name="run-2",
        model=MagicMock(),
        params={},
        metrics={},
        tags={"stage": "dev"},
    )

    mock_mlflow.set_tags.assert_called_once_with({"stage": "dev"})
