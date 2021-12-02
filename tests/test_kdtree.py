import numpy as np
import pytest
from src.utils import KDTree
from sklearn.neighbors import KDTree as skTree

n_neighbors = 5
n = 30
d = 5
N = 10


# генерируем случайные датасеты
@pytest.mark.parametrize("X", [np.random.randn(n, d) for _ in range(N)])
# случайные точки, для которых надо найти соседей
@pytest.mark.parametrize("X_query", [np.random.randn(n, d) for _ in range(N)])
# случайный размер листа
@pytest.mark.parametrize("leaf_size",
                         [np.random.randint(5, 20) for _ in range(3)])
def test_kd_tree(X, X_query, leaf_size):
    """
    Проверяем работу нашей реализации KDTree для случайных начальных данных
    """
    myTree = KDTree(X, leaf_size)   # наше дерево
    myNeighbors = myTree.query(X_query, k=n_neighbors)    # индексы соседей

    realTree = skTree(X, leaf_size, 'euclidean')    # дерево из библиотеки
    # возвращает координаты и индексы
    _, realNeighbors = realTree.query(X_query, k=n_neighbors)
    assert (myNeighbors == realNeighbors).all()
