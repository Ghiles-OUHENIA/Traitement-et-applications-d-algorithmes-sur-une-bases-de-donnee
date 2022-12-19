# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd


# ------------------------
def crossval(X, Y, n_iterations, iteration):
    X = list(X)
    Y = list(Y)
    Xtest = X[int(iteration * (len(X) / n_iterations)): int((iteration + 1) * (len(X) / n_iterations))]
    Ytest = Y[int(iteration * (len(Y) / n_iterations)): int((iteration + 1) * (len(Y) / n_iterations))]
    if iteration == 0:
        Xapp = X[int((iteration + 1) * (len(X) / n_iterations)):]
        Yapp = Y[int((iteration + 1) * (len(Y) / n_iterations)):]
    elif iteration == n_iterations - 1:
        Xapp = X[0:int(iteration * (len(X) / n_iterations))]
        Yapp = Y[0:int((iteration) * (len(Y) / n_iterations))]
    else:
        Xapp = X[0:int((iteration) * (len(X) / n_iterations))] + X[int((iteration + 1) * (len(X) / n_iterations)):]
        Yapp = Y[0:int((iteration) * (len(Y) / n_iterations))] + Y[int((iteration + 1) * (len(Y) / n_iterations)):]
    return np.asarray(Xapp), np.asarray(Yapp), np.asarray(Xtest), np.asarray(Ytest)


# ------------------------

def crossval_strat(X, Y, n_iterations, iteration):
    X1 = X[Y == 1]
    Y1 = Y[Y == 1]
    X_1 = X[Y == -1]
    Y_1 = Y[Y == -1]
    Xapp1, Yapp1, Xtest1, Ytest1 = crossval(list(X1), list(Y1), n_iterations, iteration)
    Xapp_1, Yapp_1, Xtest_1, Ytest_1 = crossval(list(X_1), list(Y_1), n_iterations, iteration)
    Xapp = np.stack((Xapp1, Xapp_1), axis=1)
    Yapp = np.stack((Yapp1, Yapp_1), axis=1)
    Xtest = np.append((Xtest1, Xtest_1))
    Ytest = np.append((Ytest1, Ytest_1))
    return np.asarray(Xapp), np.asarray(Yapp), np.asarray(Xtest), np.asarray(Ytest)


# ---------------------------

def analyse_perfs(perfs):
    """
    Renvoie un tuple contenant la moyenne et la variance de la précision du modèle
    """
    return (np.mean(perfs), np.var(perfs))