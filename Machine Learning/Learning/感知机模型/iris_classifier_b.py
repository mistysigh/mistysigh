# -*- coding: utf-8 -*-
"""
@Project ：mistysigh 
@File    ：iris_classifier_b.py
@IDE     ：PyCharm 
@Author  ：Sig-M
@Date    ：Created on 2023/3/1 21:03 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classification_simple

df = pd.read_csv("iris.csv", header=None)
# select setosa and versicolor
y = df.iloc[0:100, 5].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [1, 3]].values

# 标准化处理
x_std = np.copy(x)
x_std[:, 0] = (x[:, 0]-x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1]-x[:, 1].mean()) / x[:, 1].std()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = classification_simple.AdalineGD(n_iter=10, eta=0.01).fit(x, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = classification_simple.AdalineGD(n_iter=10, eta=0.0001).fit(x, y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()