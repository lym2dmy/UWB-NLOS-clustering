# -*- coding: utf-8 -*-
"""
评价clustering的效果
有别于classification和regression
"""

import os
import time
import scipy
from random import shuffle
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans,MiniBatchKMeans,AffinityPropagation,SpectralClustering,Birch,DBSCAN
from sklearn.decomposition import PCA, KernelPCA
from scipy.signal import medfilt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# Functions to determine precision, recall, F1-score and ARI
# ------------------------------------------------------------

# Get precicion
def getPrecision(mat, k, s, total):
    sum_k = 0
    for i in range(k):
        max_s = 0
        for j in range(s):
            if mat[i][j] > max_s:
                max_s = mat[i][j]
        sum_k += max_s
    return sum_k / total


# Get recall
def getRecall(mat, k, s, total, unclassified):
    sum_s = 0
    for i in range(s):
        max_k = 0
        for j in range(k):
            if mat[j][i] > max_k:
                max_k = mat[j][i]
        sum_s += max_k
    return sum_s / (total + unclassified)


# Get ARI
def getARI(mat, k, s, N):
    t1 = 0
    for i in range(k):
        sum_k = 0
        for j in range(s):
            sum_k += mat[i][j]
        t1 += scipy.special.binom(sum_k, 2)

    t2 = 0
    for i in range(s):
        sum_s = 0
        for j in range(k):
            sum_s += mat[j][i]
        t2 += scipy.special.binom(sum_s, 2)

    t3 = t1 * t2 / scipy.special.binom(N, 2)

    t = 0
    for i in range(k):
        for j in range(s):
            t += scipy.special.binom(mat[i][j], 2)

    ari = (t - t3) / ((t1 + t2) / 2 - t3)
    return ari


# Get F1-score
def getF1(prec, recall):
    return 2 * prec * recall / (prec + recall)


def performance_report(trues, preds):
    gold_standard_n_clusters = 0
    all_gold_standard_clusters_list = trues
    gold_standard_clusters_list = list(set(all_gold_standard_clusters_list))
    gold_standard_n_clusters = len(gold_standard_clusters_list)
    print("\nNumber of clusters available in the gold standard: ", gold_standard_n_clusters)
    # Get the gold standard
    # ----------------------------
    gold_standard_clusters = [[] for x in range(gold_standard_n_clusters)]
    gold_standard_count = 0
    index_trues = 0
    for row in trues:
        gold_standard_count += 1
        index_trues += 1
        contig = index_trues
        bin_num = gold_standard_clusters_list.index(row)
        gold_standard_clusters[bin_num].append(contig)

    print("Number of objects available in the gold standard: ", gold_standard_count)

    # Get the number of clusters from the initial clustering result
    # ---------------------------------------------------------
    n_clusters = 0

    all_clusters_list = []

    for row in preds:
        all_clusters_list.append(row)

    clusters_list = list(set(all_clusters_list))
    n_clusters = len(clusters_list)

    print("Number of clusters available in the clustering result: ", n_clusters)

    # Get initial clustering result
    # ----------------------------
    clusters = [[] for x in range(n_clusters)]

    clustered_count = 0
    clustered_objects = []

    index_preds = 0
    for row in preds:
        clustered_count += 1
        index_preds += 1
        contig = index_preds
        bin_num = clusters_list.index(row)
        clusters[bin_num].append(contig)
        clustered_objects.append(contig)

    print("Number of objects available in the clustering result: ", len(clustered_objects))

    # Determine precision, recall, F1-score and ARI for clustering result
    # ------------------------------------------------------------------

    total_clustered = 0

    clusters_species = [[0 for x in range(gold_standard_n_clusters)] for y in range(n_clusters)]

    for i in range(n_clusters):
        for j in range(gold_standard_n_clusters):
            n = 0
            for k in range(clustered_count):
                if clustered_objects[k] in clusters[i] and clustered_objects[k] in gold_standard_clusters[j]:
                    n += 1
                    total_clustered += 1
            clusters_species[i][j] = n

    print("Number of objects available in the clustering result that are present in the gold standard:", total_clustered)

    my_precision = getPrecision(clusters_species, n_clusters, gold_standard_n_clusters, total_clustered)
    my_recall = getRecall(clusters_species, n_clusters, gold_standard_n_clusters, total_clustered, (gold_standard_count - total_clustered))
    my_ari = getARI(clusters_species, n_clusters, gold_standard_n_clusters, total_clustered)
    my_f1 = getF1(my_precision, my_recall)

    print("\nEvaluation Results:")
    print("Precision =", my_precision)
    print("Recall =", my_recall)
    print("F1-score =", my_f1)
    print("ARI =", my_ari)
    # print("accuracy = ", cluster_acc(trues, preds))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(trues, preds))
    print("Completeness: %0.3f" % metrics.completeness_score(trues, preds))
    print("V-measure: %0.3f" % metrics.v_measure_score(trues, preds))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(trues, preds))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(trues, preds))

# if __name__ == '__main__':
#     pass