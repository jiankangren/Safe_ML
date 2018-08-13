from sklearn.base import BaseEstimator
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from Score_Func import *
from base import XData

from sklearn.metrics import accuracy_score
import numpy as np
IS_EMPTY = -1
TO_ZERO=1e-08
CRITERIA_CLF = {"gini":GiniFunc}


# class DecisionTreeClassifier(BaseEstimator):
#     """Base class for decision trees.
#     Warning: This class should not be used directly.
#     Use derived classes instead.
#     """
#     def __init__(self,
#                  criterion,
#                  splitter,
#                  max_depth,
#                  min_samples_split,
#                  min_samples_leaf,
#                  min_weight_fraction_leaf,
#                  max_features,
#                  max_leaf_nodes,
#                  random_state,
#                  min_impurity_decrease,
#                  min_impurity_split,
#                  class_weight=None,
#                  presort=False):
#         self.criterion = criterion
#         self.splitter = splitter
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_features = max_features
#         self.random_state = random_state
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.min_impurity_split = min_impurity_split
#         self.class_weight = class_weight
#         self.presort = presort

#     def fit(self, X, y, sample_weight=None, check_input=True,
#             X_idx_sorted=None):

class Tree(object):
    """
    Docstring for Tree
    Attributes
    -------------------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.
    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`
    feature :   np.array([int])
        feature[i] denotes the split  index for node i
    threshold : np.array([float]) 
        threshold_left[i]:denotes the split threshold s for node i
    children_left :  np.array([int])
        children_left[i] holds the node id of the left child of node i.
    children_right :  np.array([int])
        children_right[i] holds the node id of the left child of node i.
    value : array 
        Contains the constant prediction value of each node.
    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.
        
        In our case, only when the space contains no data, impurity is 0, 
        and the builder can stop.
    Examples
         0
       1    2
      3 4  5 6
    7
    feature :   np.array([0,1,1,2,1,0,1])
    children_left  [1,3,5,7,-1,-1,-1,-1]
    children_right  [2,4,6,-1,-1,-1,-1,-1]
    threshold  [0.5,0.19,1,3...]


    """
    def __init__(self, arg):
        super(Tree, self).__init__()
        self.children_left=[]
        self.children_right=[]
        self.children_feature=[]
        self.impurity =[]
        self.value =[]



    def depth_fist_build(self, Xdata, criterion='gini', min_samples_split=2, min_samples_leaf=1, max_depth=10, 
                        min_impurity_split=0):
        """
        min_samples_leaf int
            The minimum number of samples required to be at a leaf node:
        min_samples_split : int
            The minimum number of samples required to split an internal node
        min_impurity_split : float,
            Threshold for early stopping in tree growth. 
            A node will split if its impurity is above the threshold, 
            otherwise it is a leaf.
            Default 0 denotes that always need to split
        min_impurity_decrease : float, optional (default=0.)

        """
        if not isinstance (Xdata, XData):
            raise ValueError("XData should be in base.XData format, got %s" % type(Xdata))
        criterion = CRITERIA_CLF[self.criterion]
        i=0 # node index



if __name__=='__main__':
    data = np.array([
    [-2.571244718,4.784783929,0],
    [-3.571244718,5.784783929,0],
    [-3.771244718,1.784783929,1],
    [-2.771244718,1.784783929,1],
    [2.771244718,1.784783929,0],
    [1.728571309,1.169761413,0],
    [3.678319846,2.81281357,0],
    [3.961043357,2.61995032,0],
    [2.999208922,2.209014212,0],
    [7.497545867,3.162953546,1],
    [9.00220326,3.339047188,1],
    [7.444542326,0.476683375,1],
    [10.12493903,3.234550982,1],
    [6.642287351,3.319983761,1]])
    X=XData(data)
    data = np.array([[1],[2],[2],[3],[3],[5],[5]])
    X=XData(data)
    print GiniFunc.score( X,i=0,s=1,n_left=1,n_right=6,n_empty=7)
