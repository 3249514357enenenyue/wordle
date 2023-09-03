import pandas as pd

# 读取Excel文件
df = pd.read_excel('data.xlsx')

# 提取第二列数据
data = df.iloc[:, 1]

# 计算标准差
std = data.std()

# 计算中位数
median = data.median()

# 计算方差
variance = data.var()
# 计算平均值
mean_value = data.mean()
# 计算偏度
skewness = data.skew()

# 计算变异系数
coeff_of_variation = (std / mean_value)

# 计算峰度
kurt = data.kurtosis()



print("标准差：", std)
print("中位数：", median)
print("方差：", variance)
print("偏度：", skewness)
print("变异系数：", coeff_of_variation)
print("峰度：", kurt)
print("平均值：", mean_value)