import random
from concurrent.futures import ProcessPoolExecutor
from DecisionTree import CLTree
import numpy as np
from Score_Func import GiniFunc
from base import *
import copy_reg
import types
import multiprocessing
from tqdm import tqdm
from Preprocess import *

CRITERIA_CLF = {"gini":GiniFunc()}


class RandomForestClassifier(object):

    """
    :param  nb_trees:       Number of decision trees to use
    :param  pec_samples:    Percentage of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """
    def __init__(self, nb_trees, pec_samples, max_workers=1,criterion='gini',\
        min_samples_leaf=1, max_depth=10,gain_ratio_threshold=0.01,max_feature_p=1,max_try_p=1):
        self.trees = []
        self.nb_trees = nb_trees
        self.pec_samples = pec_samples
        self.max_workers = max_workers
        self.criterion=criterion
        self.min_samples_leaf=min_samples_leaf
        self.max_depth=max_depth
        self.gain_ratio_threshold=gain_ratio_threshold
        self.max_feature_p=max_feature_p
        self.max_try_p=max_try_p
        self.f_tree=float(0)



    """
    Trains self.nb_trees number of decision trees.
    Parameter:
    ----------
        X:   A list of lists samples to predict
    """
    def fit(self, X):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            rand_fts = map(lambda x: np.array(random.sample(X, int(len(X)*self.pec_samples))),
                           range(self.nb_trees))
            # print rand_fts
            self.trees = list(executor.map(self.train_tree, rand_fts))

    """
    Trains a single tree and returns it.
    """
    def train_tree(self, X):
        tree = CLTree(self.criterion,\
        self.min_samples_leaf,self.max_depth,self.gain_ratio_threshold)
        tree.fit(X,  self.max_feature_p,  self.max_try_p)
        return tree

    """
    Returns predictions based on ith tree and density_threshold
    """
    def tree_predict(self,i,X,density_threshold):
           return  self.trees[i].predict(X,density_threshold)

    """
    Returns predictions for a list of samples based on the votes. 
    The result is the average vote of base estimators
    
    Parameter:
    ----------
        p  (e.g., [0.1,0.2.] )  a list of probability used to determine density_threshold
    """
    def predict(self, X,p):
        density_threshold = map(lambda x:x.ICDF_threshold(p), self.trees)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            predictions=list(executor.map(self.tree_predict,range(self.nb_trees),\
                [X]*self.nb_trees ,density_threshold))
        predictions=np.array(predictions)
        # print predictions
        return np.sum(predictions, axis=0)/self.nb_trees





def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


if __name__ == '__main__':
    import pandas as pd
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import time
    from math import floor, ceil
    n=400
    from sklearn.datasets import make_moons, make_circles, make_classification
    X, y = make_classification(n_samples=n,n_features=3, n_redundant=0, n_informative=3,
                               random_state=1, n_clusters_per_class=2)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from math import floor, ceil
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    moon=make_moons(n_samples=n,noise=0.3, random_state=0)
    circle=make_circles(n_samples=n,noise=0.2, factor=0.5, random_state=1)  
    X,y=    circle

    file1=['./data/robot/','2.csv']
    X_train, X_test, y_train, y_test = read_data(file1[0])
    X=np.concatenate((X_test,X_train), axis=0)
        
    h=0.005
    x1_min, x1_max = X.T[0].min() - .1, X.T[0].max() + .1
    x2_min, x2_max =X.T[1].min() - .1, X.T[1].max() + .1
    xx, yy = np.meshgrid(np.arange(x1_min-1, x1_max+1, h),np.arange(x2_min-1, x2_max+1, h))
    
    rf=RandomForestClassifier(nb_trees=4, pec_samples=0.4, max_workers=8,\
        criterion='gini', min_samples_leaf=1, max_depth=25,gain_ratio_threshold=1e-10,\
                               max_feature_p=1,max_try_p=1)
    start_time=time.time()
    rf.fit(X)
    print 'Training Time',  time.time()-start_time
    start_time=time.time()
   
    Z= rf.predict(np.c_[xx.ravel(), yy.ravel()],0.03)
    print  'Predicting Time',  time.time()-start_time

    # Z=Z>0.15+np.zeros(len(Z))
    Z=Z.reshape(xx.shape)
    # store_output(Z,'rf_robot2_1_Z')
    # store_output(xx,'rf_robot2_1_xx')
    # store_output(yy,'rf_robot2_1_yy')

    
    ax = plt.subplot(1, 1, 1)
    cm = plt.cm.tab20c
    ax.set_title("")
    ax.scatter(X.T[0], X.T[1], cmap=cm,edgecolors='k')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(range(int(floor(x1_min)),int(ceil(x1_max))))
    ax.set_yticks(range(int(floor(x2_min)),int(ceil(x2_max))))
    cntr1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    cbar0 = plt.colorbar( cntr1,)
    plt.tight_layout()
    plt.show()
