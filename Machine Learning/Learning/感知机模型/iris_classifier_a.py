# -*- coding: utf-8 -*-
"""
@Project ：ML-and-DL-master 
@File    ：iris_classifier_a.py
@IDE     ：PyCharm 
@Author  ：Sig-M
@Date    ：Created on 2023/2/28 16:13 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classification_simple
from classification_simple import Perceptron, plot_decision_regions

df = pd.read_csv("iris.csv", header=None)
print(df.tail())

# select setosa and versicolor
y = df.iloc[0:100, 5].values
# print(y)
y = np.where(y == 'Iris-setosa', -1, 1)
# print(y)

x = df.iloc[0:100, [1, 3]].values

# plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Iris-setosa')
# plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Iris-versicolor')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

# ppn = Perceptron(eta=0.1, n_iter=30)
# ppn.fit(x, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=5)
ppn.fit(x, y)
plot_decision_regions(x, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()