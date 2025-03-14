from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


def load_data_heavy(csv_path): # Load dataset from a CSV file
    df = pd.read_csv(f"/home/{os.environ.get('USER_NAME')}/audio_deepfake_detector/processed_data/music_preprocessed.csv")
    return df

class SVCModel_rbf:

    def __init__(self, kernel='rbf', gamma=1.0):
        self.model = SVC(kernel=kernel, gamma=gamma)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def cross_val_score(self, X_test, y_test, gamma=1.0, scoring='accuracy'):
        scores = cross_val_score(self.model, X_test, y_test, gamma=gamma, scoring='accuracy')
        return np.mean(scores)

class SVCModel_poly:
    def __init__(self, kernel='poly', degree=3, epsilon = 0.1, C=1):
        self.model = SVC(kernel='poly', degree=degree, epsilon = epsilon, C=C)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def cross_val_score(self, X_test, y_test,degree=3, epsilon = 0.1, C=1, scoring='accuracy'):
        scores = cross_val_score(self.model, X=X_test, y=y_test, degree=degree, epsilon=epsilon, C=C scoring='accuracy')
        return np.mean(scores)

class SVCModel_linear:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def cross_val_score(self, X_test, y_test, cv=5, scoring='accuracy'):
        scores = cross_val_score(self.model, X=X_test, y=y_test, cv=cv, scoring='accuracy')
        return np.mean(scores)

class KNNModel :
    def __init__(self, n_neighbors = 3):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def cross_val_score(self, X_test, y_test, n_neighbors=3,scoring='accuracy'):
        scores = cross_val_score(self.model, X=X_test, y=y_test, n_neighbors=3, scoring='accuracy')
        return np.mean(scores)
