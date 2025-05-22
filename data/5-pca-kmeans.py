# -*- coding: utf-8 -*-
"""
2022年6月29日17:45:17
使用Kernel PCA降维诊断数据，再KMeans无监督机器学习模型输出结果
"""
import os
import time
import scipy
from random import shuffle
import pandas as pd
import joblib
import matplotlib
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
import sys
sys.path.append("../000tools")
from evaluate_clustering import performance_report
import seaborn as sns
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from random import randint
from collections import Counter
plt.style.use('../000tools/paperlym.mplstyle')# custom style
plt.rcParams.update({'font.size': 16, "figure.figsize": "5, 4"}) # custom font size
from sklearn.preprocessing import normalize,minmax_scale
from matplotlib.ticker import MultipleLocator,AutoMinorLocator

# warnings.filterwarnings('ignore')
configs = {'file_url':"data/nlos.csv",'save_model_url':'data/autoencoder.pth' ,'save_kmeans_model_url':'data/autoencoder_kmeans.joblib' , 'batch_size':32,'epochs':10,'lr':0.001,'max_len':14,'num_clusters':3}
num_clusters = configs['num_clusters']


def colum_distribution(da,index,is_debug=False):
    """
    统计各列的数据分布
    :param da:
    :return:
    """
    # da.hist(index, bins=50, alpha=0.5)
    col = da.iloc[:,index-1].tolist()
    if is_debug:
        print(np.min(col),np.max(col), np.mean(col), np.std(col), stats.skew(col), stats.kurtosis(col))
        plt.figure()
        plt.hist(col, bins=50, facecolor='g', alpha=0.75)
    a = np.mean(col) + np.std(col) * 4
    b = np.mean(col) - np.std(col) * 4
    c = np.array(col)
    c[(c >= a) | (c <= b)] = np.mean(col)
    # c.fillna(c.median(), inplace=True)
    print(np.min(c),np.max(c), np.mean(c), np.std(c), stats.skew(c), stats.kurtosis(c))
    if is_debug:
        plt.figure()
        plt.hist(c, bins=50, facecolor='g', alpha=0.75)
        plt.show()
    return c


def statistics_dataset_timedomain(data1, data2,data3):
    # 统计各列的数据分布
    num_bins = 50
    datas = [data1, data2,data3]
    colors = ['green','red','blue']
    # colors = ['#008176','#c1272d','#0000a7']
    labels = ['LOS','H-NLOS','S-NLOS',]
    for k, (d, c, l) in enumerate(zip(datas, colors, labels)):
        plt.figure(figsize=[10,2.5])
        for i in range(data1.shape[1]):
        # for i in [2,13]:
            di = d.iloc[:,i].values.tolist()
            num_sampling = 200
            plt.scatter(range(num_sampling),di[:num_sampling],label=(i+1),alpha=1,s=5)
        plt.xlabel("Sampling points")
        plt.ylabel("Normalized values")
        # plt.yticks(plt.yticks()[0],plt.yticks()[0]/num_bins)
        # plt.legend()
        plt.savefig("figs/timedomain-%s.png" % l, bbox_inches='tight')
        # plt.show()
        plt.close()

def statistics_dataset_together(original_data_,data1_, data2_,data3_):
    statistics = original_data_.describe()
    statistics.to_csv("data/statistics.csv")
    print(statistics)

    # 要先归一化，不然数值大小不同，难以比较
    original_data = original_data_.copy()
    data1 = data1_.copy()
    data2 = data2_.copy()
    data3 = data3_.copy()

    original_data.columns = range(original_data.shape[1])
    data1.columns = range(data1.shape[1])
    data2.columns = range(data2.shape[1])
    data3.columns = range(data3.shape[1])

    mins = original_data.min().values
    maxs = original_data.max().values
    for i in range(original_data.shape[1]):
        original_data[i] = (original_data[i]-mins[i])/(maxs[i] - mins[i])
        data1[i] = (data1[i]-mins[i])/(maxs[i] - mins[i])
        data2[i] = (data2[i]-mins[i])/(maxs[i] - mins[i])
        data3[i] = (data3[i]-mins[i])/(maxs[i] - mins[i])

    # 开始统计
    datas = [data1, data2,data3,original_data]
    colors = ['green','red','blue','gray']
    # colors = ['#008176','#c1272d','#0000a7']
    labels = ['LOS','H-NLOS','S-NLOS','All']

    # 统计不同类型下的数据分布
    for k, (d, c, l) in enumerate(zip(datas, colors, labels)):
        plt.figure(figsize=[12,2.5])
        sns.violinplot(data=d, orient='v',linewidth=1)
        plt.xticks(range(14),['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14'])
        plt.xlabel("Feature")
        plt.ylabel("Normalized values")
        plt.savefig("figs/violinplot-%s.png" % l, bbox_inches='tight')
        # plt.show()
        plt.close()
    for k, (d, c, l) in enumerate(zip(datas, colors, labels)):
        plt.figure(figsize=[12,2.5])
        sns.boxplot(data=d, orient='v',linewidth=1, width=0.3, fliersize=1, whis=2, showfliers = False)
        plt.xlabel("Feature")
        plt.ylabel("Normalized values")
        plt.xticks(range(14),['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14'])
        # plt.xticks(range(14),range(1,15))
        plt.savefig("figs/boxplot-%s.png" % l, bbox_inches='tight')
        # plt.show()
        plt.close()

    # 统计各列的数据分布
    num_bins = 50
    for i in range(data1.shape[1]):
        print(i)
        plt.figure()
        for d,c,l in zip(datas,colors,labels):
            di = d.iloc[:,i].values.tolist()
            ax = sns.histplot(di, stat='probability',kde=True, bins=num_bins, alpha=0.7,label=l,color=c)
            # plt.plot([0],[0],color=c,label="%s (kernel density)" % l)
            # n, bins, patches = plt.hist(di, num_bins, density=True, alpha=0.7,color=c,label=l)
        plt.xlabel("Normalized values")
        plt.ylabel("Probability")
        # plt.yticks(plt.yticks()[0],plt.yticks()[0]/num_bins)
        plt.legend()
        # plt.show()
        plt.savefig("figs/distribution-feature-%d.png" % i, bbox_inches='tight')
        plt.close()


def statistics_dataset_after_pca(data, tmp,is_debug=False):
    # 统计各列的数据分布
    data = pd.DataFrame(data)
    data.hist(bins=50, alpha=0.5)

    # 统计相关性
    matrix = np.abs(data.corr().round(2))
    plt.figure(figsize=[6, 4.5])
    labels = [r'$y_a$', r'$y_b$', r'$y_c$']
    ax = sns.heatmap(matrix, annot=True, cmap="icefire", xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 16, "family": "Times New Roman", 'weight': 'normal'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.savefig("figs/data-corr-after-pca.png", bbox_inches='tight')
    if is_debug:
        plt.show()

    data['wall_type'] = tmp
    data1 = data[data["wall_type"] == 0]
    data2 = data[data["wall_type"] == 1]
    data3 = data[data["wall_type"] == 2]
    data1.drop(["wall_type"],inplace=True,axis=1)
    data2.drop(["wall_type"],inplace=True,axis=1)
    data3.drop(["wall_type"],inplace=True,axis=1)
    # 归一化
    # data1 = data1.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    # data2 = data2.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    # data3 = data3.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    # 统计各列的数据分布
    num_bins = 50
    datas = [data1, data2,data3]
    colors = ['green','red','blue']
    # colors = ['#008176','#c1272d','#0000a7']
    labels = ['LOS','H-NLOS','S-NLOS',]
    for i in range(data1.shape[1]):
        print(i)
        plt.figure()
        for d,c,l in zip(datas,colors,labels):
            di = d.iloc[:,i].values.tolist()
            ax = sns.histplot(di, stat='probability',kde=True, bins=num_bins, alpha=0.7,label=l,color=c)
            # plt.plot([0],[0],color=c,label="%s (kernel density)" % l)
            # n, bins, patches = plt.hist(di, num_bins, density=True, alpha=0.7,color=c,label=l)
        plt.xlabel("Normalized values")
        plt.ylabel("Probability")
        # plt.yticks(plt.yticks()[0],plt.yticks()[0]/num_bins)
        plt.legend()
        # plt.show()
        plt.savefig("figs/distribution-feature-pca-%d.png" % i, bbox_inches='tight')
        if is_debug:
            plt.show()
        else:
            plt.close()



def statistics_dataset(original_data,is_debug=False):
    # original_data.drop(["wall_type","stdNoise","maxGrowthCIR","first_ratio","LDE_PPINDX","rxPreamCount","LDE_PPAMPL"],inplace=True,axis=1)
    # original_data.drop(["wall_type"],inplace=True,axis=1)
    # 统计各列的数据分布
    if is_debug:
        original_data.hist(bins=50, alpha=0.5)
        # print("min","max", "avg std sk kurtsis")
        # for i in range(original_data.shape[1]):
        #     c = colum_distribution(original_data,i+1)
        #     original_data.iloc[:,i] = c
        # colum_distribution(original_data, 6)
        # colum_distribution(original_data, 7)
        # original_data = original_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
        matrix = np.abs(original_data.corr().round(2))
        # matrix = original_data.corr().round(2)
        # print(matrix)
        plt.figure(figsize=[12, 10])
        plt.rcParams["font.family"] = "Times New Roman"
        sns.set_style({'font.size': 12})
        print(matplotlib.get_cachedir(), matplotlib.font_manager.weight_dict)
        del matplotlib.font_manager.weight_dict['roman']
        matplotlib.font_manager._rebuild()

        labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']
        # labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
        ax = sns.heatmap(matrix, annot=True, cmap="YlOrBr", xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "family": "Times New Roman", 'weight': 'normal'})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        # use matplotlib.colorbar.Colorbar object
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=12)

        plt.savefig("data/data-corr.png", bbox_inches='tight')
        plt.show()


def pca_kernel_compare(pca_data,tmp,kernels,is_debug=False):
    for k in kernels:
        print(k)
        pca(pca_data, tmp, kernel=k, is_debug=is_debug)

def pca(pca_data_original,tmp,kernel='cosine',is_debug=False):
    # df.drop(["stdNoise","maxGrowthCIR","first_ratio","LDE_PPINDX","rxPreamCount","LDE_PPAMPL"],inplace=True,axis=1)
    if kernel=='sigmoid':
        # sigmoid层需要归一化处理，参考：https://www.zhihu.com/question/263874165 为什么sigmoid的输出全部是0.0或者1.0？ sigmoid函数的输出范围是0~1，当这一层输入值的绝对值过大时，输出会趋于饱和，观察sigmoid图像可以看出来，输入的范围应该在-6~6之间，你这种情况应该是隐层的输出的绝对值过大，导致饱和。【【【我认为可以增加归一化层（参考BN相关知识）】】】
        pca_data_original = minmax_scale(pca_data_original,axis=0)

    if kernel == 'linear':
        pca_data = PCA(n_components=3).fit_transform((pca_data_original))
    else:
        pca_data = KernelPCA(n_components=3, kernel=kernel, n_jobs=-1).fit_transform(pca_data_original)

    # 准备数据，准备绘图3d
    data1 = []
    data2 = []
    data3 = []
    for t, d in zip(tmp, pca_data):
        if t==0:
            data1.append(d)
        elif t == 1:
            data2.append(d)
        elif t == 2:
            data3.append(d)

    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)

    fig = plt.figure(figsize=[8,7])
    ax = fig.add_subplot(projection='3d')
    p = ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], color='g', marker='o',edgecolors="k",s=50, alpha=0.5, label="LOS")
    p = ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='r', marker='h',edgecolors="k",s=50, alpha=0.5, label="H-NLOS")
    p = ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], color='b', marker='s',edgecolors="k",s=50, alpha=0.5, label="S-NLOS")
    # plt.legend()
    ax.set_xlabel(r'$y_a$')
    ax.set_ylabel(r'$y_b$')
    ax.set_zlabel(r'$y_c$')
    # # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    # Bonus: To get rid of the grid as well:
    # ax.grid(True,which='both',color='b', linestyle='-.')

    # ax.set_xticks(np.arange(-1,1,0.25))
    # ax.set_yticks(np.arange(-0.6,1,0.2))
    # ax.set_zticks(np.arange(-0.6,1.0,0.2))

    # ax.xaxis.set_major_locator(MultipleLocator(0.2))
    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax.zaxis.set_major_locator(MultipleLocator(0.2))

    plt.savefig("figs/pca-kernel-%s.png" % kernel, bbox_inches='tight')
    if is_debug:
        plt.show()
    else:
        plt.close()

    return pca_data

def plot_result(pca_data, trues, preds):
    if pca_data.shape[1] ==2:
        plt.figure()
        plt.title("True")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=trues, alpha=0.5)
        plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6, 7])
        plt.figure()
        plt.title("Predicted")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=preds, alpha=0.5)
        plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6, 7])
        plt.show()
    elif pca_data.shape[1] ==3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p=ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=trues, alpha=0.5)
        plt.title("True")
        fig.colorbar(p,ticks=[0, 1, 2, 3, 4, 5, 6, 7])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p=ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=preds, alpha=0.5)
        plt.title("Predicted")
        fig.colorbar(p,ticks=[0, 1, 2, 3, 4, 5, 6, 7])
        plt.show()


def kmeans_algorithm(pca_data,tmp,n_clusters=3,is_debug=False):
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=n_clusters)
    kmeans_model = kmeans.fit(pca_data)
    joblib.dump(kmeans_model, configs['save_kmeans_model_url'], compress=3)
    yy = kmeans_model.predict(pca_data)
    performance_report(tmp, yy)
    if is_debug:
        plot_result(pca_data, tmp, yy)


def birch_algorithm(pca_data,tmp,n_clusters=3,is_debug=False):
    bir = Birch(n_clusters=n_clusters,threshold=0.01)
    bir = bir.fit(pca_data)
    yy_bir = bir.predict(pca_data)
    performance_report(tmp, yy_bir)
    if is_debug:
        plt.figure()
        c = bir.subcluster_centers_
        plt.plot(pca_data[:, 0], pca_data[:, 1], '+')
        plt.plot(c[:, 0], c[:, 1], 'o')
        plot_result(pca_data, tmp, yy_bir)
        plt.show()


def DBSCAN_algorithm(pca_data, labels_true,is_debug=False):
    ## 确定超参数 https://www.reneshbedre.com/blog/dbscan-python.html
    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros)
    nbrs = NearestNeighbors(n_neighbors=7,n_jobs=-1).fit(pca_data)
    print(nbrs.algorithm,nbrs.effective_metric_,nbrs.effective_metric_params_)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(pca_data)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    # Now, get the sorted kth column (distances with kth neighbors) and plot the kNN distance plot,
    k_dist = sort_neigh_dist[:, 4]

    # # 二次差分法寻找knee
    # kkx = range(len(k_dist))
    # kky = k_dist
    # from scipy import interpolate
    # uspline = interpolate.interp1d(kkx, kky)
    # kky2 = uspline(kkx)
    # kky2 = np.array(kky2)
    # kky2 = kky2[:-100]
    # k_dist2 = np.diff(kky2, n=2)
    # plt.plot(kky2)
    # plt.plot(k_dist2)
    # print(k_dist[np.argmax(k_dist2)+1],np.argmax(k_dist2), len(k_dist2))
    # 调包，寻找knee
    kneedle = KneeLocator(range(len(k_dist)), k_dist, curve="convex", direction="increasing")
    print("调包，寻找knee",kneedle.elbow_y)
    print("曲线的均值和标准差情况",np.mean(k_dist), np.std(k_dist))
    elbow_y = kneedle.elbow_y
    # if kneedle.elbow_y is None:
    #     elbow_y = 5
    #     print("调包，寻找knee",elbow_y)

    if is_debug:
        plt.figure()
        plt.plot(k_dist,color='blue',linewidth=2)
        # 确定拐点是0.015
        # plt.axhline(y=k_dist[np.argmax(k_dist2)+1], linewidth=1, linestyle='dashed', color='k')
        # plt.axhline(y=kneedle.elbow_y, linewidth=2, linestyle='dashed', color='r')
        plt.scatter([kneedle.elbow],[kneedle.elbow_y],color='r',s=100)
        # plt.axhline(y=0.015, linewidth=1, linestyle='dashed', color='k')
        plt.ylabel("k-NN distance")
        plt.xlabel("Sorted observations (4th NN)")
        plt.show()

    # 开始DBSCAN
    # db = DBSCAN(eps=k_dist[np.argmax(k_dist2)+1], min_samples=7, n_jobs=-1).fit(pca_data)
    db = DBSCAN(eps=elbow_y, min_samples=27, n_jobs=-1).fit(pca_data) # 之前min_samples=7分类出6种类型，其中最后两个类型的个数均小于27，所以此处我改成了27，这样就只分成了3类+异常类（共计4类）。注意：每类的个数可以在print(Counter(db.labels_))这里看到。
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(set(db.labels_))
    print(Counter(db.labels_))

    # output results
    performance_report(labels_true, db.labels_)
    if is_debug:
        plot_result(pca_data, labels_true, db.labels_)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # plot the results
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        colors = []
        for i in range(len(unique_labels)):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                # col = [0, 0, 0, 1]
                col = 'black'

            class_member_mask = labels == k

            xy = pca_data[class_member_mask & core_samples_mask]
            p = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2],
                c=col,
                edgecolors="k",
                s=50, alpha=0.5)
            # plt.plot(
            #     xy[:, 0],
            #     xy[:, 1],
            #     "o",
            #     markerfacecolor=tuple(col),
            #     markeredgecolor="k",
            #     markersize=14,
            # )

            xy = pca_data[class_member_mask & ~core_samples_mask]
            # plt.plot(
            #     xy[:, 0],
            #     xy[:, 1],
            #     "o",
            #     markerfacecolor=tuple(col),
            #     markeredgecolor="k",
            #     markersize=6,
            # )
            p = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2],
                c=col,
                edgecolors="k",
                s=20, alpha=0.5)

        plt.title("Estimated number of clusters: %d" % n_clusters_)

        plt.show()

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_data, labels))


def get_data():
    data = pd.read_csv(configs['file_url'], index_col=0)
    print("集大小", data.shape)
    # print(data)
    le = LabelEncoder()
    new_class = le.fit_transform(data['wall_type'])
    data.loc[:,'wall_type'] = new_class
    print(le.classes_)
    # df = data.copy()

    n_clusters = 3
    ## 注意此处！！！！！！！！！！！！！！！
    ## 注意此处！！！！！！！！！！！！！！！
    # 0-man; 1-wall; 2-window; 3-black; 4-brick; 5-concrete; 6-los; 7-wood
    # 015；347；2；6是los但是略微像347
    # 0-man；2-glass Windows；6-los
    df = data.loc[(data["wall_type"] == 0) | (data["wall_type"] == 2) | (data["wall_type"] == 6)]
    print(df.groupby("wall_type").size())
    for i in range(df.shape[0]):
        if df.iloc[i,0]==0:
            df.iloc[i,0] = 3
    for i in range(df.shape[0]):
        if df.iloc[i,0]==3:
            df.iloc[i,0] = 1
        elif df.iloc[i,0]==6:
            df.iloc[i,0] = 0
    print(df.groupby("wall_type").size())

    tmp = np.array(df['wall_type'].tolist())

    data1 = df[df["wall_type"] == 0]
    data2 = df[df["wall_type"] == 1]
    data3 = df[df["wall_type"] == 2]
    data1.drop(["wall_type"], inplace=True, axis=1)
    data2.drop(["wall_type"], inplace=True, axis=1)
    data3.drop(["wall_type"], inplace=True, axis=1)
    df.drop(["wall_type"], inplace=True, axis=1)
    pca_data_original = np.array(df.values.tolist())

    return data, df,pca_data_original,tmp,data1,data2,data3


def main():
    pass



if __name__ == '__main__':
    print("============================begin")
    n_clusters = 3
    is_debug = False
    is_debug = True
    data, df,pca_data_original,tmp,data1,data2,data3 = get_data()

    # 统计原始数据集
    # statistics_dataset(df,is_debug=is_debug)
    # statistics_dataset_timedomain(data1, data2,data3)
    # statistics_dataset_together(df,data1, data2,data3)

    # 不降维直接聚类
    # DBSCAN_algorithm(pca_data_original, tmp) # 0.97，如果不自动寻找knee的话，精度为0.7甚至更低
    # kmeans_algorithm(pca_data_original, tmp, n_clusters=n_clusters) # 0.76
    # birch_algorithm(pca_data_original, tmp, n_clusters=n_clusters) # 0.92

    # 比较不同的降维kernel效果
    kernels = ['cosine','linear','rbf','poly','sigmoid']
    # kernels = ['cosine','linear','poly']
    # poly 时参数：,gamma=0.2,degree=2
    # linear 参数：,eigen_solver='dense',max_iter=10
    # pca_kernel_compare(pca_data_original,tmp,kernels,is_debug=is_debug)

    # 使用上一步对比的最优的核函数进行降维
    pca_data = pca(pca_data_original,tmp,kernel=kernels[1],is_debug=is_debug)
    # 统计降维后数据集
    # statistics_dataset_after_pca(pca_data, tmp,is_debug=is_debug)
    # print("\nKMeans")
    # kmeans_algorithm(pca_data, tmp, n_clusters=n_clusters) #
    # print("\nBirch")
    # birch_algorithm(pca_data, tmp, n_clusters=n_clusters)
    # print("\n DBSCAN")
    DBSCAN_algorithm(pca_data, tmp,is_debug=is_debug)

    print("============================end")