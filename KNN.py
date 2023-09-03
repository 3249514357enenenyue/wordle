import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# 加载Excel文件，并提取"Number of reported results"列
data = pd.read_excel("test.xlsx")
results = data["Number of reported results"]

# 创建训练集和测试集
X_train = results[:-30].values.reshape(-1, 1)
y_train = results[30:].values

# 训练KNN模型
knn = KNeighborsRegressor(n_neighbors=12)
knn.fit(X_train, y_train)

# 预测未来30天的结果
future_results = []
current_results = results.iloc[-30:].values.reshape(-1, 1)
for _ in range(31):
    next_result = knn.predict(current_results)
    future_results.append(next_result[0])
    current_results = np.append(current_results[1:], next_result[0].reshape(-1, 1), axis=0)

# 输出预测结果
predicted_data = pd.DataFrame({"Number of reported results": future_results})
predicted_data.to_csv("out.csv", index=False)
