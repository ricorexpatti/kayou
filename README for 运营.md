## 🛠️ 使用准备

### 1. 下载并安装 Anaconda
- 进入 [Anaconda 下载页面](https://www.anaconda.com/download)  
- 选择你的操作系统（Windows / Mac / Linux）  
- 一路点击 **下一步 / Continue** 安装（保持默认设置即可）  

👉 安装好后，你的电脑里就已经有 **Python + Jupyter Notebook + 必要库** 了。  

---

### 2. 打开 Jupyter Notebook
1. 打开 **Anaconda Navigator**  
2. 点击 **Jupyter Notebook → Launch**  
3. 浏览器会自动打开
4. 找到并点击项目文件 `卡游库存预测模型.py`  

---

### 3. 运行代码
- 在 Jupyter Notebook 里，你会看到一格一格的代码（Cell）  
- 用鼠标点选第一格，然后点击 **▶ Run**（或键盘按 **Shift + Enter**）  
- 一格一格往下运行，运行结果会显示在下面  

---
---
## ✏️ 需要修改什么？

大多数情况下，你只需要修改 数据路径 和 预测天数。

### 数据文件路径
在代码里找到：
```python
df = pd.read_csv("卡牌发货.csv", encoding="utf-8")
```
---
- 把 "卡牌发货.csv" 改成你自己的 CSV 文件名路径。
- 如果是 Excel 文件，改成：
```python
pd.read_excel("你的文件.xlsx")
```
---
### 预测天数
在166行代码里找到：
```python
period = 30
window_out = 30
```
---
- 默认预测 30 天,可改成 7天、60天等，但请确保你的数据长度足够查询。

### 列名对应
确保你的数据文件里，有一列叫 `日期`，另一列叫 `wms实际发货数量`。
- 如果名字不同，改成你文件里的实际列名。
- 例如：
```python
df = df.rename(columns={"发货量": "wms实际发货数量"})
```
---
## 📊 输出结果
运行后可看到：
- 趋势拟合曲线（预测 vs 实际对比）
- 模型预测量（未来 30 天的发货量）
- 误差指标（告诉你预测是否准确：MAE、RMSE、MAPE）

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
