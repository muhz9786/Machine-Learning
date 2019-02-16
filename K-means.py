from sklearn.cluster import k_means as Kmeans
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

# K-Means模型,比较不同K值时的聚类效果
score = {}
for N in range(2, 8):
    print('K=%s' % N)
    y, lables, inertia = Kmeans(x, n_clusters=N, random_state=1)
    CH = metrics.calinski_harabaz_score(x, lables)
    print(CH)
    score[N] = CH

# 评价指标可视化
plt.plot(range(2, 8), score.values(), marker='o')
plt.show()

# 最终聚类结果
N = max(score, key=score.get)
start = time.perf_counter()
y, lables, inertia = Kmeans(x, n_clusters=N, random_state=1)
end = time.perf_counter()
print('Running time: %s ms' % ((end - start) * 1000))
for i in range(N):
    print('第 {0} 类有 {1} 个样本'.format(i, np.where(lables == i)[0].size))
print('轮廓系数：{}'.format(metrics.silhouette_score(x, lables)))

# 聚类结果可视化
c0 = np.where(lables == 0)
c1 = np.where(lables == 1)
c2 = np.where(lables == 2)
c3 = np.where(lables == 3)
ax = plt.subplot(111, projection='3d')
ax.scatter(x[c0, 0], x[c0, 1], x[c0, 2], c='y')
ax.scatter(x[c1, 0], x[c1, 1], x[c1, 2], c='r')
ax.scatter(x[c2, 0], x[c2, 1], x[c2, 2], c='g')
ax.scatter(x[c3, 0], x[c3, 1], x[c3, 2], c='b')
plt.show()
