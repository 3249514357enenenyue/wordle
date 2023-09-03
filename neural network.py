import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 从 Excel 文件中读取数据
data = pd.read_excel('test.xlsx')

# 提取日期和报告结果列
dates = data['Date'].values
reported_results = data['Number of reported results'].values

# 数据预处理
x_train = np.arange(1, len(dates) + 1).reshape(-1, 1)
y_train = reported_results.reshape(-1, 1)

# 转换为 Tensor
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# 构建神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, features):
        super(NeuralNetwork, self).__init__()

        self.linear_relu1 = nn.Linear(features, 256)
        self.linear_relu2 = nn.Linear(256, 512)
        self.linear_relu3 = nn.Linear(512, 512)
        self.linear_relu4 = nn.Linear(512, 1024)  # 新增的线性层
        self.linear5 = nn.Linear(1024, 1)  # 新增的线性层

    def forward(self, x):
        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)  # 使用新增的线性层
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear5(y_pred)
        return y_pred

model = NeuralNetwork(features=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss(reduction='mean')

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 预测未来30天的结果
x_test = np.arange(len(dates) + 1, len(dates) + 32).reshape(-1, 1)
x_test_tensor = torch.from_numpy(x_test).float()
predicted_results_tensor = model(x_test_tensor)
predicted_results = predicted_results_tensor.detach().numpy().flatten()

# 创建结果DataFrame
result = pd.DataFrame({'日期': dates[-1] + pd.to_timedelta(np.squeeze(x_test), unit='D'),
                       '预测结果': predicted_results})

# 将结果保存为CSV文件
result.to_csv('output.csv', index=False)
