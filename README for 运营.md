# 🛠️ 使用准备

## 1. **安装 Python**  
   - 到 [Python 官网](https://www.python.org/downloads/) 下载并安装最新版（建议 3.10+）。  
   - 安装时记得勾选 **Add to PATH**。

## 2. 安装 `Jupyter` 或者 `pycharm`
```bash
pip install notebook ```

在项目目录里运行：
```bash
jupyter notebook ```

## ✏️ 需要修改什么？

大多数情况下，你只需要修改 数据路径 和 预测天数。

## 数据文件路径
在代码里找到：
```
df = pd.read_csv("卡牌发货.csv", encoding="utf-8") ```
- 把 "卡牌发货.csv" 改成你自己的 Excel/CSV 文件名（文件放在同一个文件夹下）。
