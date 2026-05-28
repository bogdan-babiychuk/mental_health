import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import model_dump, model_load, train


@pytest.fixture
def simple_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])


def test_train_returns_fitted_gridsearch(classification_xy, simple_pipeline):
    X, y = classification_xy
    grid = {"clf__C": [0.1, 1.0]}

    gs = train(X, y, simple_pipeline, grid)

    assert isinstance(gs, GridSearchCV)
    assert hasattr(gs, "best_estimator_")
    assert gs.best_params_["clf__C"] in [0.1, 1.0]
    assert 0.0 <= gs.best_score_ <= 1.0


def test_train_can_predict(classification_xy, simple_pipeline):
    X, y = classification_xy
    gs = train(X, y, simple_pipeline, {"clf__C": [1.0]})

    preds = gs.predict(X)
    assert len(preds) == len(y)


def test_model_dump_and_load_roundtrip(tmp_path, classification_xy, simple_pipeline):
    X, y = classification_xy
    simple_pipeline.fit(X, y)
    path = tmp_path / "model.pkl"

    model_dump(simple_pipeline, str(path))
    assert path.exists()

    loaded = model_load(str(path))
    assert (loaded.predict(X) == simple_pipeline.predict(X)).all()


def test_model_dump_invalid_path_raises(simple_pipeline):
    with pytest.raises(Exception, match="Не удалось загрузить модель"):
        model_dump(simple_pipeline, "/nonexistent_dir_xyz/model.pkl")
