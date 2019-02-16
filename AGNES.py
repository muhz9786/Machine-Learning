from sklearn.cluster import AgglomerativeClustering as AGNES
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 读取数据
f = pd.read_csv("./data/Walking2.csv", names=['d', 'x', 'y', 'z'])
x = f.iloc[:, 1:4]

# 归一化（使用极差法）
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# 数据集可视化
ax = plt.subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2],  marker='o')
plt.show()

# AGNES模型,比较不同K值时的聚类效果
for linkage in ('ward', 'average', 'complete'):
    print('linkage= ' + linkage)
    start = time.perf_counter()
    y = AGNES(n_clusters=4, linkage=linkage).fit(x)
    end = time.perf_counter()
    print('Running time: %s ms' % ((end - start) * 1000))
    for i in range(4):
        print('第 {0} 类有 {1} 个样本'.format(i, np.where(y.labels_ == i)[0].size))
    score = (metrics.silhouette_score(x, y.labels_))
    print('轮廓系数：{}'.format(score))

    # 聚类结果可视化
    c0 = np.where(y.labels_ == 0)
    c1 = np.where(y.labels_ == 1)
    c2 = np.where(y.labels_ == 2)
    c3 = np.where(y.labels_ == 3)
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x[c0, 0], x[c0, 1], x[c0, 2], c='y')
    ax.scatter(x[c1, 0], x[c1, 1], x[c1, 2], c='r')
    ax.scatter(x[c2, 0], x[c2, 1], x[c2, 2], c='g')
    ax.scatter(x[c3, 0], x[c3, 1], x[c3, 2], c='b')
    plt.show()
