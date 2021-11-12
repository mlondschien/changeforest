from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

_LETTERS_FILE = "letters.csv"
_LETTERS_PATH = Path(__file__).parent.resolve() / _LETTERS_FILE

_IRIS_FILE = "iris.csv"
_IRIS_PATH = Path(__file__).parent.resolve() / _IRIS_FILE

def load_letters():
    if _LETTERS_PATH.exists():
        return pd.read_csv(_LETTERS_PATH)
    else:
        dataset = fetch_openml(data_id=6)["frame"]
        dataset.to_csv(_LETTERS_PATH)
        return dataset

def load_iris():
    if _IRIS_PATH.exists():
        return pd.read_csv(_IRIS_PATH)
    else:
        dataset = fetch_openml(data_id=61)["frame"]
        dataset.to_csv(_IRIS_PATH)
        return dataset