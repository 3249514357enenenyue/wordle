import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

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
plt.savefig('clustering_result.png')  # 保存聚类结果图片
plt.show()

# 假设输入的数据为input_data，是一个包含 '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', 'X tries' 列的 DataFrame
input_data = pd.DataFrame([['0.0022', '0.9556', '4.7941', '23.0558', '35.3386', '27.1365', '7.2198']], columns=['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', 'X tries'])

# 使用PCA进行降维处理（与之前的代码一致）
reduced_input_data = pca.transform(input_data)

# 使用训练好的KMeans模型进行预测
predicted_label = kmeans.predict(reduced_input_data)

# 输出预测的类别
print("输入数据属于类别: ", predicted_label[0])