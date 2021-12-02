import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score, accuracy_score
from src.utils import get_precision_recall_accuracy

max_class = 5
n = 20
N = 10


@pytest.mark.parametrize("y_true",
                         [np.random.randint(0, max_class, n) for _ in range(N)]
                         )
@pytest.mark.parametrize("y_pred",
                         [np.random.randint(0, max_class, n) for _ in range(N)]
                         )
class TestClass:
    def test_precision_score(self, y_true, y_pred):
        assert (
            precision_score(
                y_true,
                y_pred,
                average=None) == get_precision_recall_accuracy(
                y_pred,
                y_true)[0]).all()

    def test_recall_score(self, y_true, y_pred):
        assert (
            recall_score(
                y_true,
                y_pred,
                average=None) == get_precision_recall_accuracy(
                y_pred,
                y_true)[1]).all()

    def test_accuracy_score(self, y_true, y_pred):
        assert accuracy_score(
            y_true, y_pred) == get_precision_recall_accuracy(
            y_pred, y_true)[2]
