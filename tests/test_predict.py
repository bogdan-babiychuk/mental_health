import numpy as np
import pytest
from unittest.mock import MagicMock

from src.predict import get_metrics, get_predict


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([0, 1, 1, 0])
    m.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.9, 0.1],
    ])
    return m


def test_get_predict_default_uses_predict(mock_model):
    X = np.zeros((4, 3))
    out = get_predict(mock_model, X)

    mock_model.predict.assert_called_once()
    assert list(out) == [0, 1, 1, 0]


def test_get_predict_return_proba(mock_model):
    X = np.zeros((4, 3))
    out = get_predict(mock_model, X, return_proba=True)

    np.testing.assert_array_equal(out, np.array([0.2, 0.7, 0.6, 0.1]))


def test_get_predict_custom_threshold(mock_model):
    X = np.zeros((4, 3))
    out = get_predict(mock_model, X, threshold=0.65)

    np.testing.assert_array_equal(out, np.array([0, 1, 0, 0]))


def test_get_predict_threshold_lower(mock_model):
    X = np.zeros((4, 3))
    out = get_predict(mock_model, X, threshold=0.15)

    np.testing.assert_array_equal(out, np.array([1, 1, 1, 0]))


def test_get_metrics_without_proba():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = get_metrics(y_true, y_pred)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert metrics["accuracy"] == pytest.approx(4 / 5)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_get_metrics_with_proba():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = get_metrics(y_true, y_pred, y_proba)

    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
