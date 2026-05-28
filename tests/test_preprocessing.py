import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, get_train_test


def test_build_preprocessor_non_tree(numeric_features, categorical_features):
    pre = build_preprocessor(numeric_features, categorical_features, tree=False)

    assert isinstance(pre, ColumnTransformer)
    transformers = dict((name, trans) for name, trans, _ in pre.transformers)
    assert isinstance(transformers["num"], Pipeline)
    assert "scaler" in dict(transformers["num"].steps)
    assert "poly" in dict(transformers["num"].steps)


def test_build_preprocessor_tree_passthrough(numeric_features, categorical_features):
    pre = build_preprocessor(numeric_features, categorical_features, tree=True)

    transformers = dict((name, trans) for name, trans, _ in pre.transformers)
    assert transformers["num"] == "passthrough"


def test_build_preprocessor_fit_transform_shape(sample_df, numeric_features, categorical_features):
    pre = build_preprocessor(numeric_features, categorical_features, tree=False)
    X = sample_df[numeric_features + categorical_features]

    out = pre.fit_transform(X)


    assert out.shape[0] == len(sample_df)
    assert out.shape[1] >= len(numeric_features) + len(categorical_features)


def test_get_train_test_split_sizes(sample_df, target_col):
    Xtr, Xte, ytr, yte = get_train_test(sample_df, target_col=target_col)

    total = len(sample_df)
    assert len(Xtr) + len(Xte) == total
    assert len(ytr) + len(yte) == total
    assert abs(len(Xte) / total - 0.2) < 0.05
    assert target_col not in Xtr.columns


def test_get_train_test_drops_columns(sample_df, target_col, drop_cols):
    Xtr, Xte, _, _ = get_train_test(sample_df, target_col=target_col, to_drop=drop_cols)

    for col in drop_cols:
        assert col not in Xtr.columns
        assert col not in Xte.columns


def test_get_train_test_is_deterministic(sample_df, target_col):
    Xtr1, _, ytr1, _ = get_train_test(sample_df, target_col=target_col)
    Xtr2, _, ytr2, _ = get_train_test(sample_df, target_col=target_col)

    pd.testing.assert_frame_equal(Xtr1, Xtr2)
    pd.testing.assert_series_equal(ytr1, ytr2)


def test_get_train_test_stratifies(sample_df, target_col):
    _, _, ytr, yte = get_train_test(sample_df, target_col=target_col)

    train_ratio = ytr.mean()
    test_ratio = yte.mean()
    assert abs(train_ratio - test_ratio) < 0.15
