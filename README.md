
# LSTM 影评情感分析模型

## 数据集介绍

本项目使用 **Amazon US Reviews** 数据集，该数据集包含来自 Amazon 的商品评论，涵盖多个类别，如：

- **图书（Books）**
- **手机电子产品（Mobile Electronics）**
- **电脑（PC）**
- **食品（Grocery）**
- **视频游戏（Video Games）**

每条评论包含以下信息：

- `review_body`：评论文本。
- `star_rating`：用户评分（1-5 星）。
- `product_category`：商品类别。

在本项目中，我们将 `star_rating` 从 **1-5** 转换为 **0-4** 作为分类标签。

##  模型结构

本项目使用 **双向 LSTM（Bidirectional LSTM）** 进行文本分类，主要结构包括：

- **嵌入层（Embedding Layer）**：将文本转换为密集向量。
- **双向 LSTM 层（Bidirectional LSTM Layer）**：捕获文本的上下文信息。
- **全连接层（Fully Connected Layer）**：将 LSTM 输出映射到分类标签。
- **Softmax 层**：将输出转换为 5 类星级评分。

### **超参数设定**

| 参数          | 取值   |
| ----------- | ---- |
| 词向量维度       | 128  |
| 隐藏层维度       | 256  |
| LSTM 层数     | 2    |
| 是否使用双向 LSTM | 是  |
| Dropout 率   | 0.3  |
| 批量大小        | 64   |
| 学习率         | 5e-4 |
| 训练轮数        | 5    |

##  训练步骤

### **1 加载数据集**

```python
from datasets import load_dataset

dataset = load_dataset("amazon_us_reviews", "Mobile_Electronics_v1_00", trust_remote_code=True)
```

### **2 数据预处理**

- 使用 `BertTokenizer` 进行文本分词。
- 将评分（1-5）转换为 0-4 的分类标签。
- 构建训练集和验证集。

```python
from transformers import BertTokenizer

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 转换评分 1-5 → 标签 0-4
def preprocess_labels(rating):
    return int(rating) - 1
```

### **3 训练 LSTM 模型**

```python
train_model(model, train_loader, val_loader, device, epochs=5, lr=5e-4, checkpoint_dir="check_point")
```

- 采用 **Adam 优化器** 和 **交叉熵损失函数（CrossEntropyLoss）**。
- 每个 epoch 保存模型 **检查点（checkpoint）**。
- 训练完成后，保存最佳模型 `best_lstm_model.pth`。

### **4 评估模型**

```python
test_model(model, test_loader, device)
```

- 计算 **F1-score** 以衡量模型性能。

### **5 导出 ONNX 模型**

```python
export_to_onnx(model, vocab_size)
```

- 将训练完成的模型转换为 **ONNX 格式**，用于高效推理。

### **6 运行 ONNX 推理**

```python
prediction = infer_onnx("lstm_model.onnx", tokenizer, "这个产品非常棒！", 100)
print(f'预测评分: {prediction+1}')
```

## 运行项目

### **训练模型**

```python
train_model(model, train_loader, val_loader, device, epochs=5, lr=5e-4, checkpoint_dir="check_point")
```

- 直接在 Jupyter Notebook 运行该代码，即可开始训练。
- 采用 **Adam 优化器** 和 **交叉熵损失函数（CrossEntropyLoss）**。
- 每个 epoch 保存模型 **检查点（checkpoint）**。
- 训练完成后，最佳模型保存在 `best_lstm_model.pth`。

### **测试模型**

```python
test_model(model, test_loader, device)
```

- 直接在 Jupyter Notebook 运行该代码，即可在测试集上评估模型。
- 计算 **F1-score** 以衡量模型性能。
- 确保 `best_lstm_model.pth` 已加载并处于评估模式。

### **运行推理**

```python
prediction = infer_onnx("lstm_model.onnx", tokenizer, "这个产品非常棒！", 100)
print(f'预测评分: {prediction+1}')
```

- 直接在 Jupyter Notebook 运行该代码，即可对输入文本进行评分预测。
- 确保已导出 `lstm_model.onnx` 并正确加载。

##结果示例

**输入评论：**

```
"这个产品非常棒！强烈推荐。"
```

**模型预测：**

```
预测评分: 5
```




