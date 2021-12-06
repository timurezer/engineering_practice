import click
import numpy as np
import matplotlib.pyplot as plt
from src.utils import KNearest, get_precision_recall_accuracy
from src.utils import Scaler, read_cancer_dataset, train_test_split


def plot_precision_recall(X_train, y_train, X_test, y_test, max_k=30, path='/home/timurezer/JupyterProjects/Engineering_practice/reports/figures'):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'{path}/{ylabel}.png')
    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)


def plot_roc_curve(X_train, y_train, X_test, y_test, max_k=30, path='/home/timurezer/JupyterProjects/Engineering_practice/reports/figures'):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k in ks:
        colors.append([k / ks[-1], 0, 1 - k / ks[-1]])
        knearest = KNearest(k)
        knearest.fit(X_train, y_train)
        p_pred = [p[0] for p in knearest.predict_proba(X_test)]
        tpr = []
        fpr = []
        for w in np.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > w else 1) for p in p_pred]
            tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
            fpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize = (7, 7))
    for tpr, fpr, c in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=c)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{path}/roc_curve.png')

@click.command()
# @click.argument("path_to_csv", type=click.Path())
# @click.argument("path_to_pre_rec", type=click.Path())
# @click.argument("path_to_roc", type=click.Path())
def draw_plots():
    scaler = Scaler()
    X, y = read_cancer_dataset(snakemake.input[0])
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    X_train = scaler.train_scale(X_train)
    X_test = scaler.scale(X_test)
    plot_precision_recall(X_train, y_train, X_test, y_test)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10)

if __name__ == "__main__":
    draw_plots()
