import numpy as np
from pathlib import Path

PROJECT_DIR = Path("D:/TA Mesin")
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]
N_FEATURES = 63
N_CLASSES = 29


def load_processed_data():
    proc_dir = PROJECT_DIR / "data/processed"
    X = np.load(str(proc_dir / "X.npy"))
    y = np.load(str(proc_dir / "y.npy"))
    classes = np.load(str(proc_dir / "classes.npy"), allow_pickle=True)
    return X, y, classes


def filter_detected(X, y):
    mask = (X != 0).any(axis=1)
    return X[mask], y[mask], mask
