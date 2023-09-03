import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# 读取数据
data = pd.read_excel('record.xlsx')

# 提取特征和目标变量
X = data[['frequency', 'repeat']]
y = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', 'X tries']]

# 初始化一个空的列表，用于存储模型预测结果
predicted_tries = []
new_sample = pd.DataFrame({'frequency': [544.75], 'repeat': [0]})
params = {
    'n_estimators': 500,     # 增加决策树的数量
    'max_depth': 10,          # 增加决策树的最大深度
    'learning_rate': 1e-4,    # 增加学习率
    'subsample': 0.7         # 减小样本子采样比例
}
# 训练每个tries次数的模型并进行预测
for i in range(y.shape[1]):
    # 提取当前tries次数的目标变量
    current_y = y.iloc[:, i]

    # 定义GBDT模型
    model = GradientBoostingRegressor()

    # 训练模型
    model.fit(X, current_y)

    # 预测新样本的tries次数
    predicted = model.predict(new_sample)

    # 将预测结果添加到列表中
    predicted_tries.append(predicted[0])

print('预测的tries次数:')
print(predicted_tries)
