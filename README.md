# 📦 发货量预测模型 (Shipping Forecast Model)

本项目使用 **SARIMA、XGBoost、LSTM** 等模型对卡牌发货量进行预测，提供误差分析与可视化，帮助业务人员更好地做库存管理与需求预估。


---

## 📚 目录

- [功能概览](#-功能概览)
- [环境依赖](#️-环境依赖)
- [数据准备](#-数据准备)
- [使用方式](#-使用方式)
- [模型说明](#-模型说明)
- [误差指标参考](#-误差指标参考)
- [注意事项](#-注意事项)

---

## 🚀 功能概览

- **数据处理**
  - 日期解析、缺失值处理
  - 按日聚合发货量
  - 转换为时间序列或回归特征  

- **趋势拟合**
  - 多项式回归拟合整体趋势  

- **时间序列预测**
  - **SARIMA 模型**
    - 自动差分检验（ADF）
    - 季节性建模（7天周期）
    - 原始 & log 变换建模
    - 误差分析（MAE、RMSE、MAPE）  

- **机器学习预测**
  - **XGBoost 模型**
    - 滑动窗口建模
    - 多步预测未来 30 天  

- **深度学习预测**
  - **LSTM 模型**
    - 数据归一化
    - 滑动窗口构造
    - 直接预测未来 30 天发货量
    - 误差分析与可视化  


## 🛠️ 环境依赖

请确保已安装以下依赖包：

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn xgboost tensorflow 

```
---

## 📂 数据准备

将原始数据保存为 `卡牌发货新品.csv`，放在目录中。（更改数据在此！）  
数据格式要求如下：

- **列名：**
  - `日期`：格式 `YYYY/MM/DD`
  - `wms实际发货数量`：整数，表示当日发货量  

---
## 📊 使用方式

复制 卡游库存预测模型-新品 的代码到的 python 本地环境运行

输出内容包含

趋势拟合曲线

1. SARIMA 预测（原始 & log 版本）

2. 模型误差分析（MAE、RMSE、MAPE）

3. LSTM 预测 vs 实际值对比图

4. 未来 30 天预测表

## 📈 模型说明

### SARIMA

适用于样本较短但具有季节性/周期性的时间序列（本项目设定 7 天季节性，适配周周期波动）。

提供原始数据建模与 log1p 变换建模两套方案。

### XGBoost （暂未使用）

使用滑动窗口构造监督学习数据，逐步预测未来 30 天。

数据量越多，越能发挥对非线性趋势的捕捉能力（默认注释，按需开启）。

### LSTM

深度学习序列模型，擅长建模长期依赖。

使用 MinMax 归一化 + 滑窗，直接输出未来 30 天预测。

## 📑 误差指标参考

### MAE（Mean Absolute Error，平均绝对误差）

单位为发货量，越低越好。

### RMSE（Root Mean Squared Error，均方根误差）

对大误差更敏感。一般 ≤ 平均发货量的 10%~20% 为较好水平。

### MAPE（Mean Absolute Percentage Error，平均绝对百分比误差）

< 10%：非常好

10% ~ 20%：可接受

20% ~ 50%：偏高，需要改进

#> 50%：参考意义有限

## 模型调用
### sarima模型调用 （step即预测周期）
```
forecast_mean, forecast_ci = sarima_predict(train_data['wms实际发货数量'], steps=30)
```

### LSTM模型调用 （window_in为训练周期，window_out为预测周期）
```values = df_grouped['wms实际发货数量'].values
lstm_predictions = lstm_predict(values, window_in=15, window_out=30)
```

### XGBOOST模型调用 （window_in为训练周期，window_out为预测周期）
```values = df_grouped['wms实际发货数量'].values
xgboost_predictions = xgboost_predict(values, window_in=15, window_out=30)
```

## 📌 注意事项

目前数据量较少，模型训练效果会受限，预测趋势过于拟合。（后续可调整）

遇到节假日、活动促销等异常波动，建议结合 业务知识 调整或加入额外特征。

随着数据量增加，优先尝试 XGBoost/LSTM 并进行超参调优与交叉验证。


建议定期滚动重训模型，保持参数与数据新鲜度。
"""

### 数据补足后可选方案
1. 调整sarima模型架构
```# 安装所需库: pip install pmdarima

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据拆分
test_size = 30 if len(df_grouped) > 100 
train_data = df_grouped.iloc[:-test_size]
test_data = df_grouped.iloc[-test_size:]

print(f"训练集: {len(train_data)}, 测试集: {test_size}")

# 自动选择最佳参数
auto_model = auto_arima(
    train_data['wms实际发货数量'],
    seasonal=True,
    m=7,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("最佳参数:", auto_model.order, auto_model.seasonal_order)

# 使用最佳参数建模
model = SARIMAX(
    train_data['wms实际发货数量'],
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    trend='n'
)

fit_model = model.fit(disp=False)
forecast = fit_model.get_forecast(steps=test_size)
``` 

2.调整LSTM模型
```# 根据数据量自动调整参数
total_samples = len(df_grouped)

if total_samples > 200:
    window_in, window_out = 30, 30
    lstm_units, batch_size, epochs = 128, 32, 100
elif total_samples > 100:
    window_in, window_out = 21, 30
    lstm_units, batch_size, epochs = 64, 16, 80
else:
    window_in, window_out = 15, 30
    lstm_units, batch_size, epochs = 32, 8, 50

# 数据准备
values = df_grouped['wms实际发货数量'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 创建数据集
X, Y = [], []
for i in range(len(scaled) - window_in - window_out + 1):
    X.append(scaled[i:i+window_in, 0])
    Y.append(scaled[i+window_in:i+window_in+window_out, 0])
X, Y = np.array(X), np.array(Y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 划分训练验证集
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=False
)

# 构建模型
model = Sequential()
model.add(LSTM(lstm_units, return_sequences=True, input_shape=(window_in, 1)))
model.add(Dropout(0.2))
model.add(LSTM(lstm_units//2))
model.add(Dropout(0.2))
model.add(Dense(window_out))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练
history = model.fit(
    X_train, Y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, Y_val),
    callbacks=[EarlyStopping(patience=20)],
    verbose=1
)

# 预测
last_input = scaled[-window_in:].reshape(1, window_in, 1)
pred_scaled = model.predict(last_input)
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
```

