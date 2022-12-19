import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy

def normalisation(data):
    df = pd.DataFrame()
    for i in data.columns:
        l = np.array(data[i], dtype=float)
        l2 = (l-np.min(l))/(np.max(l)-np.min(l))
        df[i] = l2
    return df

def dist_euclidienne(v1, v2):
    return np.linalg.norm(v1-v2)

def dist_manhattan(v1, v2):
    return sum(abs(v1-v2))

#def dist_vect(type, v1, v2):
#    """
#    Retourne la distance euclidienne ou de manhattan entre les 2 vecteurs.
#    :param type: type de distance
#    :param v1: Vecteur 1
#    :param v2: Vecteur 2
#    :return: Number
#    """
#    if type.lower() == 'manhattan':
#        return dist_manhattan(v1, v2)
#    elif type.lower() == 'euclidienne':
#        return dist_euclidienne(v1, v2)
#    else:
#        raise "Type de distance incorrect"

def dist_vect(v1, v2):
    """
    Calcule la distance euclidienne entre v1 et v2
    """
    return dist_euclidienne(v1, v2)

def centroide(data):
    return data.mean(axis=0)

def dist_centroides(v1, v2):
    """
    Retourne la distance euclidienne entre les centroïdes des groupes de vecteurs passés en argument
    :param v1: Groupe de vecteur 1
    :param v2: Groupe de vecteur 2
    :return: Number
    """
    return dist_euclidienne(centroide(v1), centroide(v2))

def initialise(data):
    """
    Rend une partition du dataframe en argument
    :param data: dataframe pandas
    :return: dict(int:list[int])
    """
    d = dict()
    for i in range(len(data)):
        d[i] = [i]
    return d

def fusionne(data, partition, verbose=False, dist_type='euclidienne', linkage="centroide"):
    """
    Fusionne les 2 clusters les plus proches
    :param data: DataFrame étudié
    :param partition: Partition des clusters jusqu'à lors
    :param verbose:
    :param dist_type: Type de distance à calculer ('euclidienne' ou 'manhattan')
    :return: tuple(Partition1, int, int, distance(entre les 2 clusters fusionnés))
    """

    indice_i = -1
    indice_j = -1
    dist = np.inf

    for i in partition.keys():
        for j in partition.keys():
            if linkage == "centroide"  and (i != j):
                if dist_vect(centroide(data.iloc[partition[i]]), centroide(data.iloc[partition[j]])) < dist:
                    indice_j = j
                    indice_i = i
                    dist = dist_vect(centroide(data.iloc[partition[i]]), centroide(data.iloc[partition[j]]))
            elif linkage == "complete"and (i != j):
                dista_b = -1
                for a in partition[i]:
                    for b in partition[j]:
                        if dist_vect((data.iloc[a]), (data.iloc[b])) > dista_b :
                            dista_b = dist_vect((data.iloc[a]), (data.iloc[b]))

                if dista_b < dist:
                    indice_j = j
                    indice_i = i
                    dist = dista_b
            elif linkage == "simplee"and (i != j):
                dista_b = 2
                for a in partition[i]:
                    for b in partition[j]:
                        if dist_vect((data.iloc[a]), (data.iloc[b])) < dista_b :
                            dista_b = dist_vect((data.iloc[a]), (data.iloc[b]))

                if dista_b < dist:
                    indice_j = j
                    indice_i = i
                    dist = dista_b
    suiv = list(partition.keys())[-1] +1

    part = partition.copy()
    part.pop(indice_i)
    part.pop(indice_j)

    if verbose:
        print("Distance minimale trouvée entre", [indice_i, indice_j], " = ", dist)

    # part[suiv] = [partition[indice_i], partition[indice_j]]
    tmp = []
    tmp.extend(partition[indice_i])
    tmp.extend(partition[indice_j])
    part[suiv] = tmp
    return part, indice_i, indice_j, dist


# def clustering_hierarchique(data):
#   """
#   Revoie une liste de listes contenant les 2 indices d'éléments fusionnés, la distance les séparant et la somme du nombre d'éléments des 2 éléments fusionnés
#   :param data: Dataframe
#   :return:
# """
#   part = initialise(data)
#   res = []
#   while len(part.keys()) >= 2:
#       part, k1, k2, dist = fusionne(data, part)
#       tmp = []
#       tmp.append(k1)
#       tmp.append(k2)
#       tmp.append(dist)
#       tmp.append(len(part[list(part.keys())[-1]]))
#
#       res.append(tmp)
#  return res


def clustering_hierarchique(data, verbose=False, dendrogramme=False, linkage="centroide"):
    """
    Revoie une liste de listes contenant les 2 indices d'éléments fusionnés, la distance les séparant et la somme du nombre d'éléments des 2 éléments fusionnés
    :param data: Dataframe
    :return:
    """
    part = initialise(data)
    res = []
    while len(part.keys()) >= 2:
        part, k1, k2, dist = fusionne(data, part, verbose=verbose, linkage=linkage)
        tmp = [k1, k2, dist, len(part[list(part.keys())[-1]])]
        res.append(tmp)

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res,
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return res

def inertie_cluster(Ens):
    centr = centroide(Ens)
    Ens = np.asarray(Ens)
    inertie = 0
    for i in range(len(Ens)):
        inertie += dist_vect(Ens[i], centr)**2

    return inertie


def inertie_globale(Base, U):
    # Calcule l'inertie globale de l'ensemble U
    inertie = 0
    Base = np.asarray(Base)
    for i in U.keys():
        inertie += inertie_cluster(Base[U[i]])
    return inertie

def nouveaux_centroides(Base,U):
    t = []
    Base = np.asarray(Base)
    for i in U.keys():
        t.append(centroide(Base[U[i]]))
    return np.asarray(t)

def affecte_cluster(Base,Centres):
    U = {i: [] for i in range(len(Centres))}
    Base = np.asarray(Base)
    for i in range(len(Base)):
        U[plus_proche(Base[i], Centres)].append(i)
    return U

def plus_proche(Exe,Centres):
    dist = np.inf
    ind = -1
    for i in range(len(Centres)):
        if dist_vect(Exe, Centres[i]) < dist:
            dist = dist_vect(Exe, Centres[i])
            ind = i
    return ind

def init_kmeans(K,Ens):
    Ens = np.asarray(Ens)
    return Ens[np.random.choice(Ens.shape[0], K, replace=False), :]

def kmoyennes(K, Base, epsilon, iter_max):
    # Initialisation des centroides
    Centroides = init_kmeans(K, Base)
    # Affectation des exemples à chaque cluster
    U = affecte_cluster(Base, Centroides)
    # Calcul de l'inertie globale
    inertie = inertie_globale(Base, U)
    print(inertie)
    for i in range(iter_max):

        # Calcul des nouveaux centroides
        Nouveaux_Centroides = nouveaux_centroides(Base, U)

        # Mise à jour des centroides et de l'affectation des exemples
        Centroides = Nouveaux_Centroides

        # Affectation des exemples à chaque cluster
        Nouveaux_U = affecte_cluster(Base, Centroides)

        # Calcul de l'inertie globale
        Nouvelle_Inertie = inertie_globale(Base, Nouveaux_U)
        print("iteration",i+1," Inertie",Nouvelle_Inertie, "Différence", f'{abs(Nouvelle_Inertie-inertie):1.4f}')

        U = Nouveaux_U

        # Test de convergence
        if abs(Nouvelle_Inertie - inertie) < epsilon:
            break

        # Mise à jour de l'inertie globale
        inertie = Nouvelle_Inertie


    return Centroides, U