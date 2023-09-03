import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
# 读取 Excel 文件
data = pd.read_excel('record.xlsx')

# 提取需要聚类的列数据
tries_data = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', 'X tries']]

# 使用PCA进行降维处理
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tries_data)

# 使用 KMeans 聚类算法进行聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(reduced_data)

# 获取聚类结果标签
labels = kmeans.labels_

# 计算每个样本到其他样本的距离矩阵
dist_matrix = pairwise_distances(reduced_data)

# 计算每个聚类的中心点
cluster_centers = kmeans.cluster_centers_
dbi = davies_bouldin_score(reduced_data, labels)
print(dbi)
# 初始化DBI值
#dbi = 0

#for i in range(len(cluster_centers)):
    #for j in range(len(cluster_centers)):
        #if i != j:
            # 计算类间距离
            #inter_cluster_distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])

            # 计算类内距离
            #intra_cluster_distance_1 = np.mean(dist_matrix[np.ix_(np.where(labels == i)[0], np.where(labels == i)[0])])
            #intra_cluster_distance_2 = np.mean(dist_matrix[np.ix_(np.where(labels == j)[0], np.where(labels == j)[0])])
            #intra_cluster_distances = (intra_cluster_distance_1 + intra_cluster_distance_2) / 2

            # 计算DBI
            #curr_dbi = intra_cluster_distances / inter_cluster_distance

            # 累加DBI
            #dbi += curr_dbi

# 计算平均DBI
#dbi /= len(cluster_centers) * (len(cluster_centers) - 1)

# 打印DBI值
#print(f"DBI: {dbi:.2f}")
# 计算轮廓系数
silhouette_avg = silhouette_score(reduced_data, labels)

# 打印轮廓系数
print(f"Average Silhouette Score: {silhouette_avg:.2f}")
# 计算 Calinski-Harabasz 分数
ch_score = calinski_harabasz_score(reduced_data, labels)

# 打印 Calinski-Harabasz 分数
print(f"Calinski-Harabasz Score: {ch_score:.2f}")
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
plt.title('K-means Clustering')
plt.savefig('k-means clustering_result.png', dpi=1000) # 保存聚类结果图片
plt.show()
