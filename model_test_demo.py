'''
Author: drlv
Email: drlv@iflytek.com
Date: 2020-12-16 09:33:36
LastEditors: drlv
LastEditTime: 2021-02-07 15:28:42
Description: demo to compare the performances of several machine learning algorithms in three datasets
--algorithms: GBDT RF Extra SVM MLP
--datasets: iris usps adult
'''
import time

from sklearn.datasets import load_iris
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def pipeline(ML_model_name, ML_model, X_train, X_test, y_train, y_test):
    start = time.time()
    model = ML_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # score = metrics.classification_report(y_test, y_pred)
    # print(score)
    acc = accuracy_score(y_test, y_pred)
    print("{}\ttime: {:.2f}s\tacc: {:.2f}%".format(ML_model_name,
                                                   time.time()-start, 100*acc))


def load_iris_data():
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=24)
    return X_train, X_test, y_train, y_test


def load_usps_data():
    import scipy.io as sio
    import scipy.sparse as sp
    dataset = sio.loadmat('usps.mat')
    X_train = dataset['Xtr']
    X_test = dataset['Xte']
    y_train = dataset['Ytr']
    y_test = dataset['Yte']
    X_train = sp.csc_matrix.toarray(X_train)
    X_test = sp.csc_matrix.toarray(X_test)
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    return X_train, X_test, y_train, y_test


def load_adult_data():
    import pandas as pd
    train = pd.read_csv('adult/adult_train.csv', header=None)
    test = pd.read_csv('adult/adult_test.csv', header=None)

    y_train = train.iloc[:, -1]
    y_train = pd.factorize(y_train)[0]
    y_test = test.iloc[:, -1]
    y_test = pd.factorize(y_test)[0]

    X = pd.concat([train, test])
    X = pd.get_dummies(X).values
    X_train = X[:train.shape[0], :]
    X_test = X[train.shape[0]:, :]
    return X_train, X_test, y_train, y_test


model_dict = {'GBDT': GradientBoostingClassifier(), 'RF': RandomForestClassifier(
), 'Extra': ExtraTreesClassifier(), 'SVM': SVC(kernel='poly'), 'MLP': MLPClassifier(max_iter=1000)}
X_train, X_test, y_train, y_test = load_adult_data()
for ML_model_name, ML_model in model_dict.items():
    pipeline(ML_model_name, ML_model, X_train, X_test, y_train, y_test)
