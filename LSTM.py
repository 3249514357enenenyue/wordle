import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import tensorflow as tf

# 读取数据
df = pd.read_excel('test.xlsx')

# 选择需要的列
data = df['Number of reported results'].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 准备训练集和测试集
train_data = scaled_data[:len(data)-30]
test_data = scaled_data[len(data)-30:]

# 准备输入和输出
def prepare_data(data, time_steps):
    X, Y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:(i+time_steps), 0])
        Y.append(data[i+time_steps, 0])
    return np.array(X), np.array(Y)

time_steps = 1
X_train, Y_train = prepare_data(train_data, time_steps)
X_test, Y_test = prepare_data(test_data, time_steps)

# 转换输入的维度 (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(Dense(1))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-4)
# 编译和训练模型
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=2000, batch_size=16)

# 预测未来30天
future_predictions = []
last_batch = X_test[-1]
for _ in range(30):
    prediction = model.predict(last_batch.reshape(1, time_steps, 1))
    future_predictions.append(prediction)
    last_batch = np.append(last_batch[1:], prediction)

# 反归一化
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 保存结果至 output.csv
output_df = pd.DataFrame({'Number of reported results': future_predictions.flatten()})
output_df.to_csv('output.csv', index=False)
