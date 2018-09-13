#!/usr/bin/python 
# -*- coding: UTF-8 -*- 
import numpy as np
from scipy import io
import pandas as pd


mat = io.loadmat('train.mat')
# print help(mat)
matrix=mat['sarcos_inv']
data=pd.DataFrame(matrix)
data.info()
# print data.head(5)
# X=data[0:27]
# Y=data[27]
# print Y.shape

