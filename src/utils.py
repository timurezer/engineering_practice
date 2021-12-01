import numpy as np
import pandas as pd
from typing import NoReturn, Tuple, List


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    df = pd.read_csv(path_to_csv).sample(frac=1)  # перемешиваем
    df.loc[df['label'] == 'M', 'label'] = 1
    df.loc[df['label'] == 'B', 'label'] = 0

    X = df.drop('label', axis=1)
    y = df['label']
    return np.array(X), np.array(y)


def train_test_split(X: np.array, y: np.array, ratio: float
                     ) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    # у нас данные уже перемешаны, просто разделим их в нужном отношении
    split_n = int(np.around(ratio * X.shape[0]))
    X_train = X[: split_n]
    y_train = y[: split_n]
    X_test = X[split_n:]
    y_test = y[split_n:]
    return X_train, y_train, X_test, y_test


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array
                                  ) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    true = sum(y_pred == y_true)  # все правильные делить на все неправильные - accuracy
    all = y_pred.shape[0]
    accuracy = true / all
    precision, recall = np.array([]), np.array([])
    # для того, чтобы посчитать метрики для каждого класса, нам нужно
    # посчитать для них TP, FN, FP
    # еще надо понять, для чего считать, то есть нам нужен массив с классами
    classes = set(np.unique(y_true)).union(set(np.unique(y_pred)))
    classes = np.array(sorted(list(classes)))
    for item in classes:
        TP = sum((y_pred == item) * (y_true == item))
        FP = sum((y_pred == item) * (y_true != item))
        FN = sum((y_pred != item) * (y_true == item))
        if TP + FP > 0:
            precision = np.append(precision, [TP / (TP + FP)])
        else:
            precision = np.append(precision, [0])
        if TP + FN > 0:
            recall = np.append(recall, [TP / (TP + FN)])
        else:
            recall = np.append(recall, [0])
    return precision, recall, accuracy


class Node:
    def __init__(self, value=None, axis=None, data=None):
        self.value = value
        self.axis = axis
        self.data = data
        self.left = None
        self.right = None

    def single_query(self, root, point, k):

        if root.left is None and root.right is None:
            neigh_dist = np.sqrt(np.sum((root.data[:, 1:] - point) ** 2, axis=1))
            neigh_index = np.argsort(neigh_dist)
            index = root.data[neigh_index][:k]
            return neigh_dist[neigh_index], index

        else:
            axis = root.axis - 1
            if root.value > point[axis]:
                neigh_dist, index = self.single_query(root.left, point, k)
                opposite = root.right

            else:
                neigh_dist, index = self.single_query(root.right, point, k)
                opposite = root.left

            if neigh_dist[-1] >= np.sqrt(np.sum(point[axis] - root.value) ** 2) or len(index) < k:
                opposite_dist, opposite_index = self.single_query(opposite, point, k)
                return merge(opposite_index, opposite_dist, index, neigh_dist, k)

            return neigh_dist, index


def merge(opposite_ind, opposite_dist, index, neigh_dist, k):
    i = j = 0
    index_merged = []
    dist_merged = []
    while (i < len(opposite_ind)) and (j < len(index)) and (i + j < k):
        if opposite_dist[i] <= neigh_dist[j]:
            index_merged.append(opposite_ind[i])
            dist_merged.append(opposite_dist[i])
            i += 1
        else:
            index_merged.append(index[j])
            dist_merged.append(neigh_dist[j])
            j += 1
    delta = k - i - j
    index_merged.extend(opposite_ind[i: i + delta])
    dist_merged.extend(opposite_dist[i: i + delta])
    index_merged.extend(index[j: j + delta])
    dist_merged.extend(neigh_dist[j: j + delta])
    return dist_merged, index_merged


class KDTree:
    def __init__(self, X, leaf_size=40):
        self.X = np.hstack([np.arange(X.shape[0]).reshape(-1, 1), X])
        self.dim = X[0].size
        self.leaf_size = leaf_size
        self.root = self.build_tree(self.X, depth=0)

    def build_tree(self, X, depth=0):

        axis = (depth % self.dim) + 1
        median = np.median(X[:, axis])
        left, right = X[X[:, axis] < median], X[X[:, axis] >= median]
        if left.shape[0] < self.leaf_size or right.shape[0] < self.leaf_size:
            return Node(data=X)
        root = Node(value=median, axis=axis)
        root.left = self.build_tree(left, depth + 1)
        root.right = self.build_tree(right, depth + 1)
        return root

    def index_extraction(self, point, k=4):
        one_point_ans = []
        point_neigh = self.root.single_query(self.root, point, k=k)
        for index in point_neigh[1]:
            one_point_ans.append(int(index[0]))
        return one_point_ans

    def query(self, X, k=4):
        res = []
        for point in X:
            ans = self.index_extraction(point, k=k)
            res.append(ans)
        return res


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def train(self, X):  # узнаем mean и std на X_train
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def scale(self, X):
        X = (X - self.mean) / self.std
        return X

    def train_scale(self, X):  # совместим train и scale для удобства
        X = self.train(X).scale(X)
        return X


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 40):
        """
        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.
        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.tree = None
        self.labels = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.
        """
        self.tree = KDTree(X, self.leaf_size)
        self.labels = y

    def one_proba(self, point):
        labels = np.sum(self.labels[point]) / self.n_neighbors
        return np.array([1 - labels, labels])

    def predict_proba(self, X: np.array):
        neigh = self.tree.query(X, k=self.n_neighbors)
        prob = []
        for point in neigh:
            curr_prob = self.one_proba(point)
            prob.append(curr_prob)
        return prob

    def predict(self, X: np.array) -> np.array:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        Returns
        -------
        np.array
            Вектор предсказанных классов.
        """
        return np.argmax(self.predict_proba(X), axis=1)
