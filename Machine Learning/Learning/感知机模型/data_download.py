# -*- coding: utf-8 -*-
"""
@Project ：mistysigh 
@File    ：data_download.py
@IDE     ：PyCharm 
@Author  ：Sig-M
@Date    ：Created on 2023/3/1 19:19 
"""
import pandas as pd

iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
iris_data.to_csv('iris.csv')