# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
#  Version de départ : Février 2022

# Import de packages externes
from cProfile import label
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import math
import graphviz as gv


def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    val, count = np.unique(Y,return_counts=True)
    return val[np.argmax(count)]

def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    if 1 in P:
        return 0.0
    H = 0
    for i in range(len(P)):
        if P[i] != 0:
            H -= P[i]*math.log(P[i])
    return H

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    val, count = np.unique(Y, return_counts=True)
    tot = sum(count)
    p = []
    for i in count:
        p.append(i/tot)
    return shannon(p)

# ---------------------------
# ------------------------ A COMPLETER :
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        #  ------------------------------

        cpt = 0
        for i in range(len(desc_set)):
            p = self.predict(desc_set[i])
            if p == label_set[i]:
                cpt += 1
        return cpt / len(desc_set)

        #  ------------------------------

    def statsOnDF(self, desc_set, label_set):
        """
        Calcule les taux d'erreurs de classification et rend un dictionnaire.
        """
        final = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0, 'Précision': 0, 'Rappel': 0}

        for i in range(len(desc_set)):
            # print(f'Tour {i} :', '')
            est = self.predict(desc_set[i])
            # print(f'predict OK')
            if est == 1:
                if label_set[i] == est:
                    final['VP'] += 1
                else:
                    final['FP'] += 1
            else:
                if label_set[i] == est:
                    final['VN'] += 1
                else:
                    final['FN'] += 1
        final['Précision'] = final['VP'] / (final['VP'] + final['FP'])
        # VP / FN
        final['Rappel'] = final['VP'] / (final['VP'] + final['FN'])
        return final


# ---------------------------
# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        # Les variables suivantes seront initialisées par train
        self.desc_set = None
        self.label_set = None

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        if self.desc_set is None or self.label_set is None:
            raise NameError("Il faut entrainer le modèle.\nAppeler la fonction train()")
        tab_distance = np.empty(len(self.desc_set))
        for i in range(len(self.desc_set)):
            tab_distance[i] = np.linalg.norm(self.desc_set[i] - x)
        tab_indice_sort = np.argsort(tab_distance)
        p = 0
        for i in range(self.k):
            if self.label_set[tab_indice_sort[i]] == 1:
                p += 1 / self.k
        return 2 * (p - 0.5)

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x) < 0:
            return -1
        return 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set


class ClassifierKNN_MC(Classifier):
    """ Classe pour représenter un classifieur multiclasse par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k, nb_classes):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
                - nb_classes : Nombre de classes à distinguer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        self.nb_classes = nb_classes
        # Les variables suivantes seront initialisées par train
        self.desc_set = None
        self.label_set = None

    def score(self, x):
        """ renvoie la classe majoritaire parmi les k plus proches voisins
            x: une description : un ndarray
        """
        if self.desc_set is None or self.label_set is None:
            raise NameError("Il faut entrainer le modèle.\nAppeler la fonction train()")
        tab_distance = np.empty(len(self.desc_set))
        for i in range(len(self.desc_set)):
            tab_distance[i] = np.linalg.norm(self.desc_set[i] - x)
        tab_indice_sort = np.argsort(tab_distance)
        p = 0
        points_a_proximite = np.zeros(self.nb_classes, dtype=int)
        for i in range(self.k):
            # On crée une liste à 0, qui représente le nombre de points de classe son indice dans la liste. On choisit ensuite l'indice du maximum.
            a = int(self.label_set[tab_indice_sort[i]])
            # print(type(a))
            # print(a)
            points_a_proximite[a] += 1
        return np.argmax(points_a_proximite)

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return self.score(x)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set


# ---------------------------
# ------------------------ A COMPLETER :
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        w = np.random.uniform(low=-1, high=1, size=(input_dimension,))
        norm = np.linalg.norm(w)
        self.w = w / norm

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        print("Pas d’apprentissage pour ce classifieur !")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return x.dot(self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 0:
            return 1
        return -1


# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.learning_rate = learning_rate
        if init == 0:
            self.w = np.zeros((input_dimension, ))
        else:
            self.w = np.random.uniform(low=0, high=1, size=input_dimension)
            self.w = (self.w * 2) - 1
            # print(self.w.shape)

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        liste_indices = [i for i in range(len(desc_set))]
        np.random.shuffle(liste_indices)
        for indice in liste_indices:
            y = np.dot(desc_set[indice], self.w)
            """print("DESC_SET : ", desc_set[indice])
            print("W : ", self.w)
            print(y)
            print(label_set[indice])"""
            y_chap = y * label_set[indice]

            if y_chap <= 0:
                # La classification est erronée
                self.w = self.w + (self.learning_rate * label_set[indice] * desc_set[indice])

    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        norm_diff = []
        i = 0
        dist = seuil + 1
        while (i < niter_max) and (dist >= seuil):
            w_tmp = self.w.copy()
            self.train_step(desc_set, label_set)
            dist = np.linalg.norm(w_tmp - self.w)
            norm_diff.append(dist)
            i = i + 1

        return norm_diff

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # if x.shape != self.w.shape:
            # print(f'x : {x}\n x shape : {x.shape}')
            # print(self.w.shape)

        return np.dot(x, self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            #print("Score OK")
            return -1
        #print("Score OK")
        return 1


# ---------------------------
# ------------------------ A COMPLETER :
class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """

    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.learning_rate = learning_rate
        if init == 0:
            # self.w = np.zeros(input_dimension) # Ou self.w = np.zeros(noyau.get_output_dim()) ?
            self.w = np.zeros(noyau.get_output_dim())
        else:
            # self.w = np.random.uniform(low=0, high=1, size=input_dimension) # Ou self.w = np.random.uniform(low=0, high=1, size=noyau.get_output_dim())
            self.w = np.random.uniform(low=0, high=1, size=noyau.get_output_dim())
            self.w = (self.w * 2) - 1
        self.noyau = noyau

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        K_desc_set = self.noyau.transform(desc_set)
        liste_indices = [i for i in range(len(desc_set))]
        np.random.shuffle(liste_indices)
        for indice in liste_indices:
            y = np.dot(K_desc_set[indice], self.w)
            y_chap = y * label_set[indice]
            if y_chap <= 0:
                # La classification est erronée
                self.w = self.w + (self.learning_rate * label_set[indice] * K_desc_set[indice])

    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        norm_diff = []
        i = 0
        dist = seuil + 1
        while (i < niter_max) and (dist >= seuil):
            w_tmp = self.w.copy()
            self.train_step(desc_set, label_set)
            dist = np.linalg.norm(w_tmp - self.w)
            norm_diff.append(dist)
            i = i + 1

        return norm_diff

    def score(self, x):
        """ rend le score de prédiction sur x
            x: une description (dans l'espace originel)
        """
        # print(np.dot(x, self.w))
        return np.dot(x, self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        K_x = self.noyau.transform(x)
        if np.mean(self.score(K_x)) < 0:
            return -1
        return 1


# ---------------------------

class ClassifierPerceptronBiais(Classifier):
    def __init__(self, input_dimension, eps, init=0):
        self.eps = eps
        if init == 0:
            self.w = np.zeros(input_dimension)
        else:
            self.w = np.random.uniform(low=0, high=1, size=input_dimension)
            self.w = (self.w * 2) - 1
        self.allw = []

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        liste_indices = [i for i in range(len(desc_set))]
        np.random.shuffle(liste_indices)
        for indice in liste_indices:
            score = self.score(desc_set[indice])  # np.dot(desc_set[indice], self.w)
            y_chap = score * label_set[indice]
            if y_chap < 1:
                # La classification est erronée
                self.w = self.w + (self.eps * (label_set[indice] - self.score(desc_set[indice])) * desc_set[indice])
                self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        norm_diff = []
        i = 0
        dist = seuil + 1
        self.allw.append(self.w)
        while (i < niter_max) and (dist >= seuil):
            w_tmp = self.w.copy()
            self.train_step(desc_set, label_set)
            dist = np.linalg.norm(w_tmp - self.w)

            norm_diff.append(dist)
            i = i + 1

        return norm_diff

    def score(self, x):
        return np.dot(x, self.w)

    def predict(self, x):
        if self.score(x) < 1:
            return -1
        return 1

    def get_allw(self):
        return self.allw


class ClassifierPerceptron_MC(Classifier):
    def __init__(self, input_dimensions, learning_rate, nb_classes, init=0, verbose=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w:
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.learning_rate = learning_rate
        if init == 0:
            w = np.zeros(input_dimensions)
        else:
            w = np.random.uniform(low=0, high=1, size=input_dimensions)
            w = (w * 2) - 1
        self.verbose = verbose

        # On crée une liste de w pour séparer les éléments de classe i des autres.
        self.list_w = np.asarray([w.copy()] * nb_classes)

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement, fais ce traitement pour toutes les classes.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        # On crée le dictionnaire des correspondances des classes (dans le cas où les classes ne seraient pas
        # des nombres entierà partir de 0.
        classes = dict(enumerate(np.unique(label_set)))
        for i in range(len(self.list_w)):
            Yi = np.where(label_set == classes[i], 1, -1)

            # faire le traitement classique du train_step.
            liste_indices = [i for i in range(len(desc_set))]
            np.random.shuffle(liste_indices)
            for indice in liste_indices:
                y = np.dot(desc_set[indice], self.list_w[i])
                y_chap = y * Yi[indice]
                if y_chap <= 0:
                    self.list_w[i] = self.list_w[i] + (self.learning_rate * Yi[indice] * desc_set[indice])

    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        norm_diff = []
        for j in tqdm(range(len(self.list_w))):
            i = 0
            dist = seuil + 1
            while (i < niter_max) and (dist >= seuil):
                w_tmp = self.list_w[j].copy()
                self.train_step(desc_set, label_set)
                dist = np.linalg.norm(w_tmp - self.list_w[j])
                norm_diff.append(dist)
                i = i + 1
        if self.verbose:
            return norm_diff

    def score(self, x):
        list_score = []
        for j in range(len(self.list_w)):
            list_score.append(np.dot(x, self.list_w[j]))
        return list_score

    def predict(self, x):
        """ rend la prediction sur x
            x: une description
        """
        return np.argmax(self.score(x))


class ClassifierMultiOAA(Classifier):
    def __init__(self, classifier):
        self.cl = classifier
        self.classifier_list: list[Classifier]
        self.classifier_list = []

    def train(self, desc_set, label_set):
        classes = dict(enumerate(np.unique(label_set)))
        self.classifier_list = [copy.deepcopy(self.cl) for _ in range(len(classes))]
        for i in range(len(self.classifier_list)):
            ytmp = np.where(label_set == classes[i], 1, -1)
            self.classifier_list[i].train(desc_set, ytmp)

    def score(self, x):
        return [np.dot(x, self.classifier_list[i].w) for i in range(len(self.classifier_list))]

    def predict(self, x):
        return np.argmax(self.score(x))

    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()



# ---------------------------
#    Classifieur ADALINE    #
# ---------------------------

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.learning_rate = learning_rate
        self.niter_max = niter_max
        if history:
            self.allw = []
        else:
            self.allw = None

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        if self.allw is None:
            for i in range(len(desc_set)):
                if i == 64:
                    print(self.w)
                grad = np.dot(desc_set[i].T, np.dot(desc_set[i], self.w) - label_set[i])
                # grad = desc_set[i].T * ((desc_set[i]*self.w) - label_set[i])
                if np.isnan(grad).any():
                    print(f"PROBLEME TOUR {i}")
                    print(desc_set[i])
                    print()
                    print(self.w)
                    print()
                    break
                else:
                    self.w = self.w - self.learning_rate*grad
        else:
            self.allw.append(self.w)
            for i in range(len(desc_set)):
                print("TOUR {i}")
                if np.isnan(desc_set[i].T).any():
                    print("PROBLEME TOUR {i}")
                    grad = np.dot(desc_set[i].T, (desc_set[i] * self.w - label_set[i]))
                    self.w = self.w - self.learning_rate*grad
                    self.allw.append(self.w)

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 1:
            return 1
        else:
            return -1




def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt
        LNoms : liste des noms de features (colonnes) de description
    """

    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    entropie_classe = entropie(Y)

    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.min  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None

        #############

        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.

        ##################
        ## COMPLETER ICI !
        ##################
        liste_entropie = []
        for j in range(len(X[0])):
            xj = X[:,j]
            val, count = np.unique(xj, return_counts=True)
            tot = sum(count)
            e = 0
            for v in range(len(val)):
                e+= (entropie(Y[xj == val[v]])*(count[v]/tot))
            liste_entropie.append(e)
        i_best = np.argmin(liste_entropie)
        gain_max = entropie_classe - liste_entropie[i_best]
        Xbest_valeurs = np.unique(X[:,i_best])
        #############

        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.

    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1
        return g


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        ##################
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
        ##################

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x
        """
        ##################
        return self.racine.classifie(x)
        ##################

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


def partitionne(mdesc, mclass, n, s):
    """
    Découpe le dataset en 2 sous dataset à partir des paramètres.
    :param mdesc: Dataset des attributs
    :param mclass: Labels des données
    :param n: numéro de colonne pour laquelle on sépare
    :param s: valeur seuil
    :return: tuple[ndarray, ndarray]
    """
    tmp = np.column_stack((mdesc, mclass))
    return (tmp[mdesc[:,n] <= s][:,:-1], tmp[mdesc[:,n] <= s][:,-1]), (tmp[mdesc[:,n] > s][:,:-1], tmp[mdesc[:,n] > s][:,-1])


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
        output: 
            - un tuple (seuil_trouve, entropie) qui donne le seuil trouvé et l'entropie associée
            - (None , +Inf) si on ne peut pas discrétiser (moins de 2 valeurs d'attribut)
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)



class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe

        if exemple[self.attribut] <= float(self.seuil):
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils['inf'].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            return self.Les_fils['sup'].classifie(exemple)

    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g



def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_set = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        liste_entropie = []
        for j in range(len(X[0])):
            xj = X[:,j]
            val, count = np.unique(xj, return_counts=True)
            tot = sum(count)
            e = 0
            for v in range(len(val)):
                e+= (entropie(Y[xj == val[v]])*(count[v]/tot))
            liste_entropie.append(e)
        # i_best = np.argmin(liste_entropie)
        i_best = np.argmax(entropie_classe - np.array(liste_entropie))
        gain_max = entropie_classe - liste_entropie[i_best]
        tmp, liste_vals = discretise(X,Y,i_best)
        # print(i_best)
        Xbest_seuil = tmp[0]
        Xbest_tuple = partitionne(X,Y, i_best,Xbest_seuil)
        # print(liste_entropie)
        ############
        
        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud


class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------
