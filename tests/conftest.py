import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "mental_health.csv"

TARGET_COL = "Depression"
DROP_COLS = ["Person_ID", "Anxiety", "Burnout"]
NUMERIC_FEATURES = [
    "Age",
    "Daily_Screen_Time",
    "Social_Media_Usage",
    "Night_Usage",
    "Sleep_Hours",
    "Stress_Level",
    "Work_Study_Hours",
    "Social_Interaction_Score",
    "Caffeine_Intake",
    "Smoking",
    "Alcohol",
]
CATEGORICAL_FEATURES = ["Gender", "Occupation", "Physical_Activity"]


@pytest.fixture(scope="session")
def full_df():
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def sample_df(full_df):
    return (
        full_df.groupby(TARGET_COL, group_keys=False)
        .sample(n=50, random_state=42)
        .reset_index(drop=True)
    )


@pytest.fixture
def numeric_features():
    return list(NUMERIC_FEATURES)


@pytest.fixture
def categorical_features():
    return list(CATEGORICAL_FEATURES)


@pytest.fixture
def target_col():
    return TARGET_COL


@pytest.fixture
def drop_cols():
    return [c for c in DROP_COLS if c != "Person_ID"]


@pytest.fixture
def classification_xy(full_df):
    df = full_df.drop(columns=DROP_COLS + CATEGORICAL_FEATURES)
    X = df.drop(columns=[TARGET_COL]).head(300)
    y = df[TARGET_COL].head(300)
    return X, y
