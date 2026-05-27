import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")
    return df