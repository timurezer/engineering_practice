import numpy as np
import pytest
from src.utils import KDTree
from sklearn.neighbors import KDTree as skTree

n_neighbors = 5
n = 30
d = 5
N = 10

@pytest.mark.parametrize("X", [np.random.randn(n, d) for _ in range(N)])    # генерируем случайные датасеты
@pytest.mark.parametrize("X_query", [np.random.randn(n, d) for _ in range(N)])    # случайные точки, для которых надо найти соседей
@pytest.mark.parametrize("leaf_size", [np.random.randint(5, 20) for _ in range(3)])     # случайный размер листа
def test_kd_tree(X, X_query, leaf_size):
    """
    Проверяем работу нашей реализации KDTree для случайных начальных данных
    """
    myTree = KDTree(X, leaf_size)   # наше дерево
    myNeighbors = myTree.query(X_query, k=n_neighbors)    # индексы соседей

    realTree = skTree(X, leaf_size, 'euclidean')    # дерево из библиотеки
    _, realNeighbors = realTree.query(X_query, k=n_neighbors)    # возвращает координаты и индексы
    assert (myNeighbors == realNeighbors).all()




