#!/usr/bin/env python
# coding: utf-8

# ### 安装包

# In[5]:


get_ipython().system('pip install xgboost')


# ### Package Import

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# ### 数据读取

# In[7]:


df = pd.read_csv('/Users/allen/Desktop/KAYOU/库存模型/卡游库存/卡牌发货新品.csv', encoding='utf-8')

# 指定格式解析
df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')

# 按日期分组并求和
df_grouped = df.groupby('日期')['wms实际发货数量'].sum().reset_index()
df_grouped = df_grouped.sort_values(by='日期')
df_grouped.set_index('日期', inplace=True)

print(f"数据总长度: {len(df_grouped)}")


# ### 数据类型趋势观察

# In[8]:




# 将 '日期' 列转换为日期格式
df['日期'] = pd.to_datetime(df['日期'])

# 按 '日期' 分组并求和 'wms实际发货数量'
df_grouped = df.groupby('日期')['wms实际发货数量'].sum().reset_index()

# 按日期排序
df_grouped = df_grouped.sort_values(by='日期')

# 将日期转换为数字
df_grouped['日期_numeric'] = (df_grouped['日期'] - df_grouped['日期'].min()).dt.days

# 数据拟合（多项式拟合）
X = df_grouped['日期_numeric'].values.reshape(-1, 1)
y = df_grouped['wms实际发货数量'].values

poly = PolynomialFeatures(degree=5)  # 使用5次多项式拟合
X_poly = poly.fit_transform(X)

# 拟合数据
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_poly, y)

# 预测拟合曲线
y_pred = model.predict(X_poly)

# 可视化拟合曲线
plt.plot(df_grouped['日期'], y, label='原始数据', color='blue')
plt.plot(df_grouped['日期'], y_pred, label='拟合曲线', color='red')
plt.title('Date vs shipping quantity fitting plot')
plt.xlabel('date')
plt.ylabel('shipping quantity')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 数据趋势非线性，且数据量较少，可采取时间序列ARIMA模型

# ## SARIMA模型（季节性7天）

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------
# 自动ADF检验 + 自动差分
# -------------------------
def get_diff_order(series, max_diff=3):
    """
    自动检测平稳性并返回合适的差分阶数 d
    """
    d = 0
    result = adfuller(series.dropna())
    print(f"初始: ADF={result[0]:.3f}, p={result[1]:.3f}")
    
    while result[1] > 0.05 and d < max_diff:
        d += 1
        series = series.diff().dropna()
        result = adfuller(series)
        print(f"差分 d={d}: ADF={result[0]:.3f}, p={result[1]:.3f}")
    
    return d

# -------------------------
# 数据拆分
# -------------------------
test_size = 30
if len(df_grouped) > test_size:
    train_data = df_grouped.iloc[:-test_size]
    test_data = df_grouped.iloc[-test_size:]
else:
    train_data = df_grouped
    test_data = None

print(f"训练集长度: {len(train_data)}")
if test_data is not None:
    print(f"测试集长度: {len(test_data)}")

# -------------------------
# 自动选择差分阶数 d
# -------------------------
d = get_diff_order(train_data['wms实际发货数量'])
print(f"最终选择的差分阶数 d={d}")

# -------------------------
# 模型1：原始数据建模
# -------------------------
model_arima_raw = SARIMAX(
    train_data['wms实际发货数量'],
    order=(1, d, 1),
    seasonal_order=(1, 0, 1, 7),
    trend='n'
)
fit_arima_raw = model_arima_raw.fit(disp=False)

forecast_raw = fit_arima_raw.get_forecast(steps=30)
forecast_raw_mean = forecast_raw.predicted_mean
forecast_raw_ci = forecast_raw.conf_int()

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['wms实际发货数量'], label='Train', marker='o')
if test_data is not None:
    plt.plot(test_data.index, test_data['wms实际发货数量'], label='Test', marker='x')
plt.plot(forecast_raw_mean.index, forecast_raw_mean, label='Forecast (next 30d)', linestyle='--', color='green')
plt.fill_between(forecast_raw_ci.index, forecast_raw_ci.iloc[:,0], forecast_raw_ci.iloc[:,1],
                 alpha=0.2, color='skyblue', label='95% CI')
plt.title('SARIMA model estimate')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 打印预测结果
print("\n===== 原始数据预测（30天） =====")
last_date = df_grouped['日期'].iloc[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
print("预测日期\t 预测数量")
for date, value in zip(forecast_dates, forecast_raw_mean):
    print(f"{date.strftime('%Y-%m-%d')}\t{value:.2f}")

# -------------------------
# 模型2：对数变换建模
# -------------------------
train_log = np.log1p(train_data['wms实际发货数量'])

model_arima_log = SARIMAX(
    train_log,
    order=(1, d, 1),
    seasonal_order=(1, 0, 1, 7),
    trend='n'
)
fit_arima_log = model_arima_log.fit(disp=False)

forecast_log = fit_arima_log.get_forecast(steps=30)
forecast_log_mean = np.expm1(forecast_log.predicted_mean)
forecast_log_ci = np.expm1(forecast_log.conf_int())

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['wms实际发货数量'], label='Train', marker='o')
if test_data is not None:
    plt.plot(test_data.index, test_data['wms实际发货数量'], label='Test', marker='x')
plt.plot(forecast_log_mean.index, forecast_log_mean, label='Forecast (next 30d)', linestyle='--', color='green')
plt.fill_between(forecast_log_ci.index, forecast_log_ci.iloc[:,0], forecast_log_ci.iloc[:,1],
                 alpha=0.2, color='skyblue', label='95% CI')
plt.title('SARIMA model estimate (log transform)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 打印预测结果
print("\n===== log变换预测（30天） =====")
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
print("预测日期\t 预测数量")
for date, value in zip(forecast_dates, forecast_log_mean):
    print(f"{date.strftime('%Y-%m-%d')}\t{value:.2f}")

# -------------------------
# 误差分析
# -------------------------
if test_data is not None:
    # 原始模型误差
    forecast_raw_test = forecast_raw_mean.iloc[:len(test_data)]
    mae_raw = mean_absolute_error(test_data['wms实际发货数量'], forecast_raw_test)
    rmse_raw = mean_squared_error(test_data['wms实际发货数量'], forecast_raw_test, squared=False)
    mape_raw = (np.abs((test_data['wms实际发货数量'] - forecast_raw_test) / test_data['wms实际发货数量'])).mean() * 100

    # log模型误差
    forecast_log_test = forecast_log_mean.iloc[:len(test_data)]
    mae_log = mean_absolute_error(test_data['wms实际发货数量'], forecast_log_test)
    rmse_log = mean_squared_error(test_data['wms实际发货数量'], forecast_log_test, squared=False)
    mape_log = (np.abs((test_data['wms实际发货数量'] - forecast_log_test) / test_data['wms实际发货数量'])).mean() * 100

    print("\n===== 模型误差分析 =====")
    print("指标\t\t 原始模型\t log模型")
    print(f"MAE \t {mae_raw:.2f}\t\t {mae_log:.2f}")
    print(f"RMSE\t {rmse_raw:.2f}\t\t {rmse_log:.2f}")
    print(f"MAPE\t {mape_raw:.2f}%\t {mape_log:.2f}%")

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data['wms实际发货数量'], label='Test value', marker='o')
    plt.plot(test_data.index, forecast_raw_test, label='model prediction', linestyle='--')
    plt.plot(test_data.index, forecast_log_test, label='log model prediction', linestyle='--')
    plt.title("Predict value vs test value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


# MAE / RMSE：绝对误差，单位是“发货数量”。
# 
# 一般来说，RMSE ≈ 数据平均水平的 10%~20%，算是不错。
# 
# MAPE：相对误差（百分比）。
# 
# <10%：非常好
# 
# 10%~20%：可接受
# 
# 20%~50%：偏高，模型还需要改进
# 
# #>50%：预测参考意义不大（除非数据本身极度波动）

# 目前数据量较少，时间序列模型表现更好一点，等数据量加大，可采取XGBOOST模型

# ## XGBOOST模型预测

# # 构造滑动窗口数据
# values = df_grouped['wms实际发货数量'].values
# window_in = 15
# window_out = 30
# 
# X, Y = [], []
# for i in range(len(values) - window_in - window_out + 1):
#     X.append(values[i:i+window_in])
#     Y.append(values[i+window_in:i+window_in+window_out])
# X, Y = np.array(X), np.array(Y)
# 
# # 拆成训练集/测试集
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
# 
# # 训练模型（逐天预测未来30天）
# models = []
# for step in range(window_out):
#     y_step = Y_train[:, step]   # 第 step 天的目标
#     model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
#     model.fit(X_train, y_step)
#     models.append(model)
# 
# # 用最后15天预测未来30天
# last_input = values[-window_in:]
# preds = []
# cur_input = last_input.copy()
# 
# for step in range(window_out):
#     pred = models[step].predict(cur_input.reshape(1, -1))[0]
#     preds.append(pred)
#     # 可以决定是否更新输入（滚动预测）
#     cur_input = np.append(cur_input[1:], pred)
# 
# # 打印预测结果
# future_dates = pd.date_range(start=df_grouped['日期'].max() + pd.Timedelta(days=1), periods=30)
# future_df = pd.DataFrame({'日期': future_dates, '预测发货量': np.array(preds, dtype=int)})
# print(future_df)
# 

# ## LSTM模型预测

# In[17]:


# 数据准备
values = df_grouped['wms实际发货数量'].values.reshape(-1, 1)

# 归一化（LSTM 训练更稳定）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

window_in = 15
window_out = 30

X, Y = [], []
for i in range(len(scaled) - window_in - window_out + 1):
    X.append(scaled[i:i+window_in, 0])
    Y.append(scaled[i+window_in:i+window_in+window_out, 0])
X, Y = np.array(X), np.array(Y)

# LSTM 需要 [样本数, 时间步, 特征数]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 搭建模型
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(window_in, 1)))
model.add(Dense(window_out))  # 直接预测30天
model.compile(optimizer='adam', loss='mse')

# 训练
model.fit(X, Y, epochs=50, batch_size=16, verbose=1)

# 用最后一个窗口预测未来30天
last_input = scaled[-window_in:].reshape(1, window_in, 1)
pred_scaled = model.predict(last_input)
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# 打印预测结果
future_dates = pd.date_range(start=df_grouped['日期'].max() + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({'日期': future_dates, '预测发货量': pred.astype(int)})
print(future_df)
if len(df_grouped) > window_out:
    test_true = df_grouped['wms实际发货数量'].iloc[-window_out:].values

    # 误差指标
    mae = mean_absolute_error(test_true, pred)
    rmse = mean_squared_error(test_true, pred, squared=False)
    mape = np.mean(np.abs((test_true - pred) / test_true)) * 100

    print("\n===== 误差分析 =====")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # 可视化对比
    plt.figure(figsize=(12,6))
    plt.plot(df_grouped['日期'].iloc[-window_out:], test_true, label='test value', marker='o')
    plt.plot(future_dates, pred, label='prediction value', marker='x', linestyle='--')
    plt.title("LSTM prediction vs test value")
    plt.xlabel("data")
    plt.ylabel("shipping quantity")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
else:
    # 如果数据不足，直接画完整数据 + 预测
    plt.figure(figsize=(12,6))
    plt.plot(df_grouped['日期'], df_grouped['wms实际发货数量'], label='historical value', marker='o')
    plt.plot(future_dates, pred, label='prediction value', marker='x', linestyle='--')
    plt.title("LSTM prediction")
    plt.xlabel("data")
    plt.ylabel("shipping quantity")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

