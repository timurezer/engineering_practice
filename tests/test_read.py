import numpy as np
import pytest
from src.utils import read_cancer_dataset
import pandas as pd

X, y = read_cancer_dataset("../data/raw/cancer.csv")
df = pd.DataFrame(data=X)

def test_Nans():
    assert np.sum(df.isnull().any(axis=1)) == 0

def text_Max():
    assert np.max(X) < 1e5

def text_Min():
    assert np.min(X) > -1e5

