from src.utils import read_cancer_dataset, train_test_split, Scaler, KNearest
import numpy as np

if __name__ == '__main__':
    scaler = Scaler()
    X, y = read_cancer_dataset("../data/raw/cancer.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    X_train = scaler.train_scale(X_train)
    X_test = scaler.scale(X_test)
    clf = KNearest(n_neighbors=5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f'accuracy: {100 * np.mean(pred == y_test) // 1}%')
