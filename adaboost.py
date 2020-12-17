import numpy as np 
import pandas as pd 
import os 
import sys 
from os import path as op 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from scipy import stats 
from tqdm import tqdm
def mode(x):
    if (len(x)==0):
        return -1
    return stats.mode(x.flatten())[0][0]

class WeightedStump():
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight):
        if np.unique(y).size <= 1:
            return
        N, D = X.shape
        w = sample_weight
        pos_weights = w[y==1]
        neg_weights = w[y==-1]
        class_vals, count = np.unique(y, return_counts=True)
        self.stump_positive_label = class_vals[np.argmax(count)]
        self.stump_negative_label = None
        self.stump_feature = None
        self.stump_threshold = None

        X = np.round(X)
        
        max_ig = float("-inf")
        for d in np.arange(D):
            for value in np.unique(X):
                positive_label_candidate = mode(y[X[:,d] > value])
                negative_label_candidate = mode(y[X[:,d] <= value])
                y_pred = positive_label_candidate * np.ones(N)
                y_pred[X[:, d] <= value] = negative_label_candidate

                # Handle the empty list condition and skip this.
                try:
                    positive_rate = np.unique(y[X[:,d] > value], return_counts=True)[1][1] / N
                except IndexError:
                    continue

                negative_rate = 1 - positive_rate
                y2 = np.asarray([np.sum(neg_weights), np.sum(pos_weights)]) / np.sum(w)

                sub_y_yes = y[X[:,d] > value]
                sub_w_yes = w[X[:,d] > value]
                sub_positive_rate = np.asarray([np.sum(sub_w_yes[sub_y_yes == -1]), np.sum(sub_w_yes[sub_y_yes == 1])])/np.sum(w)
                sub_negative_rate = 1 - sub_positive_rate

                ig = (stats.entropy(y2)) - ((positive_rate * 
                            stats.entropy(sub_positive_rate))) - ((negative_rate * stats.entropy(sub_negative_rate)))
                
                if ig > max_ig:
                    max_ig = ig
                    self.stump_feature = d
                    self.stump_threshold = value
                    self.stump_positive_label = positive_label_candidate
                    self.stump_negative_label = negative_label_candidate
        return 

    def predict(self, X):
        M, _ = X.shape
        X = np.round(X)

        if self.stump_feature is None:
            exit("Error! None value stump_feature encountered!")

        y_pred = np.zeros(M)
        y_pred[X[:, self.stump_feature] <= self.stump_threshold] = self.stump_negative_label
        y_pred[X[:, self.stump_feature] > self.stump_threshold] = self.stump_positive_label

        return y_pred

class AdaBoost():
    def __init__(self, n_estimators = 10):
        self.n_estimators = n_estimators
        self.estimators= None
        self.alphas = None
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        w = np.full(n_samples, (1 / n_samples))
        self.estimators = []
        self.alphas = []
        for _ in range(self.n_estimators):
            sklearn_stump = WeightedStump()
            sklearn_stump.fit(X,y,w)
            predictions = sklearn_stump.predict(X)
            accuracy = accuracy_score(y, predictions, sample_weight=w)
            error = 1 - accuracy
            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.estimators.append(sklearn_stump)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, 1))
        for i, clf in enumerate(self.estimators):
            predictions = np.expand_dims(clf.predict(X),1)
            y_pred += self.alphas[i] * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred

    def score(self, X, y):
        preds = self.predict(X)
        labels = np.squeeze(y)
        assert preds.shape == labels.shape
        return np.mean(preds == labels)

def adaboost(dataset:str) -> None:
    # Read the data and replace question mark values with nans and then impute nans with ones.
    df = pd.read_csv(dataset, sep=',' , header=None).replace('?', np.nan).fillna(value=1).astype('int32')
    y = df[10].to_numpy()
    # Encode labels. 
    y[y==2] = 1 
    y[y==4] = -1 
    X = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].to_numpy().astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    # Adaboost sanity check
    # sk_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100)
    # sk_clf.fit(X_train,y_train)
    # print(sk_clf.score(X_test,y_test))
    
    tr_errors = []
    test_errors = []
    for n_estimators in tqdm(np.arange(1, 101)):
        model = AdaBoost(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        tr_error = 1.0-model.score(X_train, y_train)
        test_error= 1.0-model.score(X_test, y_test)
        tr_errors.append(tr_error)
        test_errors.append(test_error)
        print("Training error with", n_estimators, "estimators:",tr_error)
        print("Test error with", n_estimators, "estimators:",test_error)

    plt.plot(np.arange(1, 101), tr_errors, 'b-',
                np.arange(1, 101), test_errors, 'r-')
    plt.title("AdaBoost training and test errors")
    plt.xlabel("number of estimators")
    plt.ylabel("Error")
    plt.legend(['training error', 'test error'])
    plt.show()
    plt.clf()
    return 


if __name__ == "__main__":
    adaboost("data/breast-cancer-wisconsin.data")