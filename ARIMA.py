import pandas as pd
import numpy as np
import statsmodels.api as sm

# 从 Excel 文件中读取数据
data = pd.read_excel('test.xlsx')

# 提取日期和报告结果列
dates = data['Date'].values
reported_results = data['Number of reported results'].values

# 数据预处理
x_train = np.arange(1, len(dates) + 1)
y_train = reported_results


# ADF测试
adf_test = sm.tsa.stattools.adfuller(y_train)
print("ADF统计量：", adf_test[0])
print("p-value：", adf_test[1])

# 构建ARIMA模型
model = sm.tsa.ARIMA(y_train, order=(1,0,1))

# 拟合模型
model_fit = model.fit()

# 预测未来30天的结果
x_test = np.arange(len(dates+1), len(dates) +31)
predicted_results = model_fit.predict(start=len(dates), end=len(dates) + 30)

# 创建结果DataFrame
result = pd.DataFrame({'日期': dates[-1] + pd.to_timedelta(x_test, unit='D'),
                       '预测结果': predicted_results})

# 将结果保存为CSV文件
result.to_csv('output.csv', index=False)
