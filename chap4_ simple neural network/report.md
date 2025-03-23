### 线性回归实现报告

#### 1. 函数定义

在本实现中，定义了以下几个关键函数：

- `synthetic_data(w, b, num_examples)`: 生成合成数据集，其中`w`和`b`是线性模型的参数，`num_examples`是样本数量。
- `data_iter(batch_size, features, labels)`: 打乱数据集并生成小批量数据。
- `linreg(X, w, b)`: 定义线性回归模型。
- `squared_loss(y_hat, y)`: 定义均方损失函数。
- `sgd(params, lr, batch_size)`: 实现小批量随机梯度下降算法。

#### 2. 数据采集

数据集通过`synthetic_data`函数生成，其中包含1000个样本，每个样本有两个特征。真实参数设置为`w = [2, -3.4]`和`b = 4.2`，噪声项服从均值为0、标准差为0.01的正态分布。

```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

#### 3. 模型描述

模型是一个简单的线性回归模型，形式为`y = Xw + b`，其中`X`是特征矩阵，`w`是权重向量，`b`是偏置项。损失函数使用均方损失，优化算法使用小批量随机梯度下降（SGD）。模型参数通过随机初始化，然后通过多次迭代更新以最小化损失函数。

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
```

#### 4. 拟合效果

通过训练，模型参数逐渐逼近真实参数。训练过程中的损失函数值逐渐减小，表明模型拟合效果良好。最终，模型参数`w`和`b`的估计误差非常小，接近于0。

```python
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

输出结果：
```
w的估计误差: tensor([ 0.0012, -0.0006], grad_fn=<SubBackward0>)
b的估计误差: tensor([-0.0001], grad_fn=<RsubBackward1>)
```
