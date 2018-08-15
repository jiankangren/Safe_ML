from sklearn.base import BaseEstimator
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score
from Score_Func import GiniFunc

IS_LEAF = -1
TO_ZERO=1e-08
CRITERIA_CLF = {"gini":GiniFunc}


class Node(object):
    """
    Structure for Tree Node,
    Each node contains the belonging training data, and corresponding rules
    Attributes
    -------------------

    """
    def __init__(self,X,n_empty=0,depth=1):
        """
        Attributes
        -------------------
        X: numpy.darray 
        data = np.array([
        [-2.571244718,4.784783929,1],
        [-3.571244718,5.784783929,0],)
            Input  data. 
        depth : int
            depth of the node
        n_empty : int:
            number of empty samplex
        """
        super(Node, self).__init__()
        if not isinstance (X,np.ndarray):
            raise TypeError("X should be in np.ndarray format, got %s" % type(X))
        if len(X)<=0:
            raise ValueError("Sample number should be than 0 , got %s" % len(X))
        self.X=X
        self.sortedX=np.sort(X.T)
        self.n_empty=n_empty
        self.__dict__['feature_limit']=[[x[0],x[-1]] for x in self.sortedX] #initialize
        self.__dict__['n_sample']=len(X)
        self.__dict__['n_feature']=len(X.T)
        self.depth=depth
    @property
    def feature_limit(self): 
        return  self.__dict__['feature_limit']
    @property
    def feature_type(self): 
        '''identify all category feature
        '''
        return None
    @property
    def n_sample(self): 
        return self.__dict__['n_sample']
    @property
    def n_feature(self): 
        return self.__dict__['n_feature']
    def split_by_index(self,i,s,n_empty_left,n_empty_right):
        #split data by feature i with value s
        left_index=self.X.T[i]<=s
        n_empty_left=max(n_empty_left,len(self.X[left_index]))
        NL=Node(self.X[left_index],n_empty_left,self.depth+1)
        right_index=np.logical_not(left_index)
        n_empty_right=max(n_empty_right,len(self.X[right_index]))
        NR=Node(self.X[right_index],n_empty_right,self.depth+1)
        return NL,NR


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
    threshold_left : np.array([float]) 
        threshold_left[i]:denotes the split threshold  for node i <=sL
    threshold_right : np.array([float]) 
        threshold_left[i]:denotes the split threshold  for node i >=sL:
    children_left :  np.array([int])
        children_left[i] holds the node id of the left child of node i.
    children_right :  np.array([int])
        children_right[i] holds the node id of the left child of node i.
    Leaf : array 
        Contains leaf nodes 
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
        self.feature=[]
        self.threshold_left=[]
        self.threshold_right=[]
        self.Leaf={}

    def depth_fist_build(self,X, criterion='gini', min_samples_leaf=1, max_depth=10):
        """
        X : np.darray
            The input data
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
        if max_depth<=1:
            raise ValueError("Tree depth should be greater than 1, get  %d" %(max_depth))
        criterion = CRITERIA_CLF[self.criterion]
        n_feature=len(X.T)
        heap=[0] # heap for construct the tree
        i=0 # node index
        root=Node(X,len(X),1)
        if  root.n_sample<min_samples_leaf:
                raise ValueError("Root should has more than %d samples, get  %d" \
                    %(min_samples_leaf,root.n_sample))
        elif  root.n_sample==min_samples_leaf or :
            print 'root.n_sample= min_samples_leaf'
            return root
        Leaf[0]=root
        while len(heap)>0:
            current_index=heap.pop()
            current_node=Leaf[current_index]

            index,s=best_spliter(current_node)

            xmin=current_node.feature_limit[index][0]
            xmax=current_node.feature_limit[index][1]
            L=float(xmax-xmin)
            n_empty_left=current_node.n_empty*((s-xmin)/L)
            n_empty_right=current_node.n_empty*((xmax-s)/L)

            child_left,child_right=current_node.split_by_index(index,s,n_empty_left,n_empty_right)
            
            self.feature[current_index]=index
            self.threshold_left[current_index]=s
            self.threshold_right[current_index]=children_right.feature_limit[i][0]

            if min(child_left.n_sample,child_right.n_sample)<=min_samples_leaf:
                print 'min'
           
            i+=1
            id_left=i
            self.childen_left[current_index]=id_left
            i+=1
            id_right=i
            self.children_right[current_index]=id_right

            if child_left.depth>=max_depth:
                self.childen_left[id_left]=IS_LEAF
                Leaf[id_left]=child_left










# if __name__=='__main__':
#     data = np.array([
#     [-2.571244718,4.784783929,0],
#     [-3.571244718,5.784783929,0],
#     [-3.771244718,1.784783929,1],
#     [-2.771244718,1.784783929,1],
#     [2.771244718,1.784783929,0],
#     [1.728571309,1.169761413,0],
#     [3.678319846,2.81281357,0],
#     [3.961043357,2.61995032,0],
#     [2.999208922,2.209014212,0],
#     [7.497545867,3.162953546,1],
#     [9.00220326,3.339047188,1],
#     [7.444542326,0.476683375,1],
#     [10.12493903,3.234550982,1],
#     [6.642287351,3.319983761,1]])
#     X=XData(data)
#