import random
from concurrent.futures import ProcessPoolExecutor
from DecisionTree import CLTree
import numpy as np
from Score_Func import GiniFunc
from base import *
import copy_reg
import types
import multiprocessing

CRITERIA_CLF = {"gini":GiniFunc()}



class RandomForestClassifier(object):

    """
    :param  nb_trees:       Number of decision trees to use
    :param  pec_samples:    Percentage of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """
    def __init__(self, nb_trees, pec_samples, max_workers=1,criterion='gini',\
        min_samples_leaf=1, max_depth=10,gain_ratio_threshold=0.01):
        self.trees = []
        self.nb_trees = nb_trees
        self.pec_samples = pec_samples
        self.max_workers = max_workers
        self.criterion=criterion
        self.min_samples_leaf=min_samples_leaf
        self.max_depth=max_depth
        self.gain_ratio_threshold=gain_ratio_threshold


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
        tree.fit(X)
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


# if __name__ == '__main__':
#     # logging.basicConfig(level=logging.INFO)
#     # test_rf()
#     # rf=RandomForestClassifier(nb_trees=5, pec_samples=0.4, max_workers=4,criterion='gini',\
#     #     min_samples_leaf=1, max_depth=10,gain_ratio_threshold=0.01)
#     # X= np.array([
#     # [-2.571244718,4.784783929],
#     # [-3.571244718,5.784783929],
#     # [-3.771244718,1.784783929],
#     # [-2.771244718,1.784783929],
#     # [2.771244718,1.784783929],
#     # [1.728571309,1.169761413],
#     # [3.678319846,2.81281357],
#     # [3.961043357,2.61995032],
#     # [2.999208922,2.209014212],
#     # [7.497545867,3.162953546],
#     # [9.00220326,3.339047188],
#     # [7.444542326,0.476683375],
#     # [10.12493903,3.234550982],
#     # [6.642287351,3.319983761]])
#     # y=np.ones(len(X))
#     # rf.fit(X)
#     # i=0
#     # # for tree in rf.trees:
#     # #     print i,'-----'
#     # #     i+=1
#     # #     print tree.density_X
#     # print rf.predict(X,0.3)

#     n=200
#     from sklearn.datasets import make_moons, make_circles, make_classification
#     X, y = make_classification(n_samples=n,n_features=2, n_redundant=0, n_informative=2,
#                                random_state=1, n_clusters_per_class=2)
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import ListedColormap
#     from math import floor, ceil
#     rng = np.random.RandomState(2)
#     X += 2 * rng.uniform(size=X.shape)
#     linearly_separable = (X, y)
#     moon=make_moons(n_samples=n,noise=0.3, random_state=0)
#     circle=make_circles(n_samples=n,noise=0.2, factor=0.5, random_state=1)  
#     X,y=    circle

#     # x11=np.linspace(0, 4, num=250)
#     # x12=np.linspace(6, 10, num=250)
#     # x1=np.concatenate((x11, x12), axis=0)
#     # x2=map(lambda x:np.sin(x)+np.random.normal(0,0.1),x1 )
#     # X=np.array([x1,x2]).T
#     # y=np.ones(len(X))

#     X= np.array([
#     [-2.571244718,4.784783929],
#     [-3.571244718,5.784783929],
#     [-3.771244718,1.784783929],
#     [-2.771244718,1.784783929],
#     [2.771244718,1.784783929],
#     [1.728571309,1.169761413],
#     [3.678319846,2.81281357],
#     [3.961043357,2.61995032],
#     [2.999208922,2.209014212],
#     [7.497545867,3.162953546],
#     [9.00220326,3.339047188],
#     [7.444542326,0.476683375],
#     [10.12493903,3.234550982],
#     [6.642287351,3.319983761]])
#     y=np.ones(len(X))


#     h=0.05
#     x1_min, x1_max = X.T[0].min() - .1, X.T[0].max() + .1
#     x2_min, x2_max = X.T[1].min() - .1, X.T[1].max() + .1
#     xx, yy = np.meshgrid(np.arange(x1_min-1, x1_max+1, h),np.arange(x2_min-1, x2_max+1, h))
#     rf=RandomForestClassifier(nb_trees=10, pec_samples=0.4, max_workers=4,\
#         criterion='gini', \
#         min_samples_leaf=1, max_depth=30,gain_ratio_threshold=0.01)
#     rf.fit(X)
#     Z= rf.predict(np.c_[xx.ravel(), yy.ravel()],0.05)
#     Z=Z.reshape(xx.shape)
#     print Z
#     ax = plt.subplot(1, 1, 1)
#     cm = plt.cm.RdBu
#     cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#     ax.set_title("Input data")
#     ax.scatter(X.T[0], X.T[1], c=y, cmap=cm_bright,edgecolors='k')
#     ax.set_xlim(x1_min, x1_max)
#     ax.set_ylim(x2_min, x2_max)
#     ax.set_xticks(range(int(floor(x1_min)),int(ceil(x1_max))))
#     ax.set_yticks(range(int(floor(x2_min)),int(ceil(x2_max))))
#     cntr1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
#     cbar0 = plt.colorbar( cntr1,)

#     plt.tight_layout()
#     plt.show()

