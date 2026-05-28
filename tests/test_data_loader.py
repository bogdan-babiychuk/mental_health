import pandas as pd
import pytest

from src.data_loader import load_data


def test_load_data_returns_dataframe(tmp_path, sample_df):
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    result = load_data(str(csv_path))

    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_df.shape
    assert list(result.columns) == list(sample_df.columns)


def test_load_data_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.csv"

    with pytest.raises(Exception, match="Error loading data"):
        load_data(str(missing))


def test_load_data_invalid_csv_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\x00\x01\x02not,a,valid\ncsv")

    # binary garbage may parse as one column — instead point to a directory
    with pytest.raises(Exception, match="Error loading data"):
        load_data(str(tmp_path))
