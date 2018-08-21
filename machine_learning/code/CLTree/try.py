import numpy as np  
import sys
from base import *
import random
import pathos.pools as pp
from concurrent.futures import ProcessPoolExecutor

import copy_reg
import types
import multiprocessing




class C(object):
    """docstring for C"""
    def __init__(self):
        super(C, self).__init__()
    def func(self,X):
        return  sum(X)


X= np.array([
    [-2.571244718,4.784783929],
    [-3.571244718,5.784783929],
    [-3.771244718,1.784783929],
    [-2.771244718,1.784783929],
    [2.771244718,1.784783929],
    [1.728571309,1.169761413],
    [3.678319846,2.81281357],
    [3.961043357,2.61995032],
    [2.999208922,2.209014212],
    [7.497545867,3.162953546],
    [9.00220326,3.339047188],
    [7.444542326,0.476683375],
    [10.12493903,3.234550982],
    [6.642287351,3.319983761]])

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


with ProcessPoolExecutor(max_workers=2) as executor:
    rand_fts = map(lambda x: random.sample(X, int(len(X)*0.3)),
                           range(2))
    c1=C()
    c2=C()
    print list(executor.map([c1.func,c2.func],  rand_fts))






