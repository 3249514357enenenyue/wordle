import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取 Excel 文件
data = pd.read_excel('record.xlsx')

# 提取需要聚类的列数据
tries_data = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', 'X tries']]

# 使用PCA进行降维处理
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tries_data)

# 使用高斯混合模型聚类算法进行聚类
gmm = GaussianMixture(n_components=3)  # 设置迭代次数和收敛条件
gmm.fit(reduced_data)

# 获取聚类结果标签
labels = gmm.predict(reduced_data)

# 计算DBI指标
dbi = davies_bouldin_score(reduced_data, labels)
print(dbi)
# 计算CH指标
ch = calinski_harabasz_score(reduced_data, labels)
print(ch)
silhouette_coef = silhouette_score(reduced_data, labels)
print(f"Silhouette Coefficient: {silhouette_coef}")

# 计算聚类结果的频数
cluster_counts = pd.Series(labels).value_counts()
# 计算聚类结果的占比百分比
cluster_percentages = cluster_counts / len(labels) * 100

# 打印每个聚类的样本个数和占比百分比
for cluster, count, percentage in zip(cluster_counts.index, cluster_counts, cluster_percentages):
    print(f"Cluster {cluster}: {count} samples ({percentage:.2f}%)")

# 绘制聚类结果图
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gaussian Mixture Model Clustering')
plt.savefig('clustering_result.png', dpi=1000)  # 设置dpi为300，提高图片清晰度
plt.show()

import statsmodels.api as sm

# 绘制 QQ 图
sm.qqplot(reduced_data[:, 0], line='s')
plt.title('QQ Plot of Principal Component 1')
plt.savefig('qq1.png', dpi=1000)
plt.show()

sm.qqplot(reduced_data[:, 1], line='s')
plt.title('QQ Plot of Principal Component 2')
plt.savefig('qq2.png', dpi=1000)
plt.show()

