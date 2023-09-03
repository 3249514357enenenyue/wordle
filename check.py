import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取数据
df = pd.read_csv('arima test.csv', header=None)
pred_values = df.iloc[:, 0].values
true_values = df.iloc[:, 1].values

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(true_values, pred_values))

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(true_values, pred_values)

# 计算均方误差（MSE）
mse = mean_squared_error(true_values, pred_values)

# 计算平均绝对百分比误差（MAPE）
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(true_values, pred_values)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Percentage Error (MAPE):", mape)
