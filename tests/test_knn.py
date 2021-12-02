import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from src.utils import KNearest
from src.utils import train_test_split


n = 100
d = 2
N = 10

# работает только в случае если 2 класса!

@pytest.mark.parametrize("X", [np.random.randn(n, d) for _ in range(N)])    # генерируем случайные датасеты
@pytest.mark.parametrize("y", [np.random.randint(0, 2, n) for _ in range(N)])    # генерируем случайные датасеты
@pytest.mark.parametrize("n_neighbors", [np.random.randint(2, 5) for _ in range(N)])
def test_knn(X, y, n_neighbors):
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=0.8)

    myClassifier = KNearest(n_neighbors, leaf_size=20)
    myClassifier.fit(X_train, y_train)
    my_pred = myClassifier.predict(X_test)

    realClassifier = KNeighborsClassifier(n_neighbors, leaf_size=20)
    realClassifier.fit(X_train, y_train)
    real_pred = realClassifier.predict(X_test)

    assert (my_pred == real_pred).all()
