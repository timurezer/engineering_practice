import numpy as np
from src.utils import train_test_split
import pytest

n = 100
d = 5
N = 10


@pytest.mark.parametrize("X", [np.random.rand(n, d) for _ in range(N)])
@pytest.mark.parametrize("y", [np.random.rand(n) for _ in range(N)])
@pytest.mark.parametrize("ratio",
                         [np.random.uniform(0.1, 0.9) for _ in range(N)])
def test_split(X, y, ratio):
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio)
    print(ratio)
    assert abs(int(X.shape[0] * ratio) - X_train.shape[0]) <= 1
    assert abs(int(y.shape[0] * ratio) - y_train.shape[0]) <= 1
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
