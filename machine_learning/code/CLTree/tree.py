from sklearn.base import BaseEstimator
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score
from Score_Func import GiniFunc
from base import *
import pudb

IS_LEAF = -1
FAIL_TO_FIND=-1
CONSTANT_SIZE=10
CRITERIA_CLF = {"gini":GiniFunc}


class Leaf(object):
    """The class of tree leaf node.

    __slots__ = ('index','left_child', 'right_child', 'feature',
    'threshold', 'purity', 'n_node_samples','score')
    """

    def __init__(self,X,n_empty=0,depth=1,feature_limit=None,split_set=set([])):
        """
        Parameters 
        -------------------
        feature_limit: [[min,max]*d]
            The boundary of the leaf node

        X: numpy.darray 
            data = np.array([
            [-2.571244718,4.784783929,1],
            [-3.571244718,5.784783929,0],)
            Input  data for different features.
        n_empty : int:
            Number of empty sample
        depth : int
            Depth of the node
        split_set  : set 
            The set of split feature index (0,1,2)
        Attributes:
        -------------------
        vol : float
            Volume of the leaf node and is equal to prod(length of each dimension)
        n_sample : int
            Number of samples
        n_feature: int 
            Number of features



        """
        super(Leaf, self).__init__()
        if not isinstance (X,np.ndarray):
            raise TypeError("X should be in np.ndarray format, got %s" % type(X))
        if len(X)<=0:
            raise ValueError("Sample number should be than 0 , got %s" % len(X))
        self.X=X
        self.sortedX=np.sort(X.T)
        self.n_empty=n_empty
        self.feature_limit=[[x[0],x[-1]] for x in self.sortedX]   if feature_limit ==None \
        else [[f[0],f[1]] for f in feature_limit]

        self.__dict__['n_sample']=len(self.X)
        self.__dict__['n_feature']=len(self.sortedX)
        self.depth=depth
        self.split_set=split_set
    @property
    def vol(self): 
        return np.prod(np.array([feature_limit[1]-feature_limit[0] \
            for feature_limit in self.feature_limit]))
    @property
    def feature_gap(self):
        return None
    @property
    def n_sample(self): 
        return self.__dict__['n_sample']
    @property
    def n_feature(self): 
        return self.__dict__['n_feature']

    def split_by_index(self,i,s,n_empty_left,n_empty_right):
        """Split data by feature i with value s"""

        left_index=self.X.T[i]<=s
        node_left=Leaf(self.X[left_index],n_empty_left,self.depth+1,\
            self.feature_limit,set(self.split_set))
        node_left.feature_limit[i][1]=s

        right_index=np.logical_not(left_index)
        node_right=Leaf(self.X[right_index],n_empty_right,self.depth+1,\
            self.feature_limit,set(self.split_set))
        node_right.feature_limit[i][0]=s

        
        return   node_left,   node_right


class Tree(object):
    """
    Class of Tree
    
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
    leaves : array 
        Contains leaf nodes 
    vol : float
        Volume of the tree  = prod(x_imax-x_imin for i)
    total_sample: int
        The number of total training samples
    root_gain: float 
        The initial score (1-p0^2-p1^2)  (- /minus ) the best score for the first split
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
    def __init__(self):
        super(Tree, self).__init__()
        self.children_left=np.zeros(CONSTANT_SIZE,dtype=int)
        self.children_right=np.zeros(CONSTANT_SIZE,dtype=int)
        self.feature=np.zeros(CONSTANT_SIZE,dtype=int)
        self.threshold_left=np.zeros(CONSTANT_SIZE)
        self.threshold_right=np.zeros(CONSTANT_SIZE)
        self.leaves={}
        self.vol=0
        self.total_sample=0
        self.root_gain=None

    # def __repr__(self):
        # str1 = ''.join(self.feature)
        # return self.str1
    def memory_update(self,size=100):
        '''Allocate more memory to array attributes
        '''
        if size<2:
            raise ValueError('MEMORY INCREASE SIZE SHOULD BE GREATER THAN 2')
        self.feature=np.append(self.feature,np.zeros(size,dtype=int))
        self.children_left=np.append(self.children_left,np.zeros(size,dtype=int))
        self.children_right=np.append(self.children_right,np.zeros(size,dtype=int))
        self.threshold_left=np.append(self.threshold_left,np.zeros(size))
        self.threshold_right=np.append(self.threshold_right,np.zeros(size))
    def decision_func(self,X):
        """Determine the leaf ID of for all X
        
        Parameter:
        ----------
        X :  [np.darray * n_predict_sample]
            Input Data
        
        Return:
        -------
        Y : [int * n_predict_sample]
            The leaf node ID each input belong to
        """
        x=0
        n_predict_sample=len(X)
        Y=np.zeros(n_predict_sample)
        for i in xrange(0,n_predict_sample):
            x=X[i]
            ID=0
            while self.feature[ID]!=IS_LEAF:
                split_index=self.feature[ID]
                if x[split_index]<=self.threshold_left[ID]:
                    ID=self.children_left[ID]
                elif x[split_index]>=self.threshold_right[ID]:
                    ID=self.children_right[ID]
                else:
                    Y[i]=-1
                    break
            if self.feature[ID]==IS_LEAF:
                Y[i]=ID 
        return Y
    def predict(self,X,inherit_data_boundary=False,threshold=0.5):
        """ Predict the extent the input data belong to confident region

        Parameter:
        ----------
        X :  [np.darray * n_predict_sample]
            Input Data
        inherit_data_boundary : boolen
            Determine whether to inherit the boundary the data in for 
            dimension that is not split during the training.
        threshold : float
            If the predict value > threshold, then x belong to the confident
            region.
        
        Return:
        -------
        Y : [float * n_predict_sample]
            0 denotes empty space and 1 denotes confident region
        """
        Y=self.decision_func(X)
        if not inherit_data_boundary:
            for i in xrange(0,len(Y)):
                if Y[i]==-1:
                    Y[i]=0
                else :
                    ID=Y[i]
                    Y[i]=self.leaves[ID].n_sample\
                    /(self.leaves[ID].n_sample+self.total_sample*self.leaves[ID].vol/self.vol)
                    
                    Y[i]=int(Y[i]>threshold)
            return np.array(Y)
        else:
            for i in xrange(0,len(Y)):
                if Y[i]==-1:
                    Y[i]=0
                else :
                    ID=Y[i]
                    split_set=self.leaves[ID].split_set
                    full_set=set(range(0,len(X.T)))
                    for j in full_set^split_set:
                        self.leaves[ID].feature_limit[j][0]=self.leaves[ID].sortedX[j][0] 
                        self.leaves[ID].feature_limit[j][1]=self.leaves[ID].sortedX[j][-1] 
                    x=X[i]
                    if_satisfy=np.prod([int(self.leaves[ID].feature_limit[j][0]<=x[j]<= self.leaves[ID].feature_limit[j][1]) \
                        for j in xrange(0,len(x))])>0
                    
                    if if_satisfy:
                        Y[i]=self.leaves[ID].n_sample\
                        /(self.leaves[ID].n_sample+self.total_sample*self.leaves[ID].vol/self.vol)
                        
                        Y[i]=int(Y[i]>threshold)
                    else:
                        Y[i]=0
            return np.array(Y)








    def depth_fist_build(self,X, criterion='gini', min_impurity_decrease=0,\
        min_samples_leaf=1, max_depth=10,threshold=0.01):
        """ Build the tree
        Parameter:
        ----------
        X : np.darray
            The input data
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node:
        min_impurity_decrease : float, optional (default=0.)
            If (ini_score - best_score after split) <  min_impurity_decrease
            stop building
            default =0
        max_depth : int 
            maximum depth of the tree to stop
        threshold: float
            If (ini_score-best_score)/root_gain < threshold , then split

        """


        if max_depth<=1:
            raise ValueError("Tree depth should be greater than 1, get  %d" %(max_depth))
        criterion = CRITERIA_CLF[criterion]
        n_feature=len(X.T)
        heap=[0] # heap for construct the tree
        i=0 # keep node index
        root=Leaf(X,n_empty=len(X),depth=1)
        self.leaves[0]=root

        '''total_sample, total volumne'''
        self.vol=root.vol
        self.total_sample=root.n_sample

        if  root.n_sample<min_samples_leaf:
                raise ValueError("Root should has more than %d samples, get  %d" \
                    %(min_samples_leaf,root.n_sample))
        elif  root.n_sample==min_samples_leaf:
            print 'root.n_sample= min_samples_leaf'
            self.feature[0]=IS_LEAF
            return 
        while len(heap)>0:
            if i>=len(self.feature)-2: #Increase the array_attributes size
                self.memory_update()            

            current_node_index=heap.pop()
            current_node=self.leaves.pop(current_node_index)
            # pudb.set_trace()

            '''update n_empty
            '''
            current_node.n_empty=max(current_node.n_sample, current_node.n_empty)

            index,s,score_gain=best_spliter(current_node,criterion,self.root_gain,
                min_impurity_decrease,min_samples_leaf,threshold)
            
            if current_node.depth==1:
                self.root_gain=score_gain
            if index==FAIL_TO_FIND:
                '''If fail to find any split, then the current node is a leaf
                '''
                self.feature[current_node_index]=IS_LEAF
                self.leaves[current_node_index]=current_node                
                continue

            xmin=current_node.feature_limit[index][0]
            xmax=current_node.feature_limit[index][1]
            L=float(xmax-xmin)

            current_node.split_set.add(index)
            
        

            n_empty_left=current_node.n_empty*((s-xmin)/L)
            n_empty_right=current_node.n_empty*((xmax-s)/L)

            node_left,node_right=current_node.split_by_index(index,s,n_empty_left,n_empty_right)

            self.feature[current_node_index]=index
            self.threshold_left[current_node_index]=node_left.feature_limit[index][1]
            # self.threshold_right[current_node_index]=node_right.feature_limit[index][0]
            self.threshold_right[current_node_index]=node_right.feature_limit[index][0]
            # 0.5*(s+node_right.feature_limit[index][0])
            
     
            i+=1
            id_left=i
            self.children_left[current_node_index]=id_left
            if   node_left.depth>=max_depth or node_left.n_sample<=min_samples_leaf:                
                self.feature[id_left]=IS_LEAF
            else:
                heap.append(id_left)
            self.leaves[id_left]=node_left
            
            i+=1
            id_right=i
            self.children_right[current_node_index]=id_right
            if   node_right.depth>=max_depth or node_right.n_sample<=min_samples_leaf:                
                self.feature[id_right]=IS_LEAF

            else:
                heap.append(id_right)
            self.leaves[id_right]=node_right



       









if __name__=='__main__':
    n=500
    from sklearn.datasets import make_moons, make_circles, make_classification
    X, y = make_classification(n_samples=n,n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from math import floor, ceil
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    moon=make_moons(n_samples=n,noise=0.3, random_state=0)
    circle=make_circles(n_samples=n,noise=0.2, factor=0.5, random_state=1)  
    X,y=  linearly_separable
    # x11=np.linspace(0, 4, num=250)
    # x12=np.linspace(6, 10, num=250)
    # x1=np.concatenate((x11, x12), axis=0)
    # x2=map(lambda x:np.sin(x)+np.random.normal(0,0.1),x1 )
    # X=np.array([x1,x2]).T




    h=0.02
    x1_min, x1_max = X.T[0].min() - .1, X.T[0].max() + .1
    x2_min, x2_max = X.T[1].min() - .1, X.T[1].max() + .1
    xx, yy = np.meshgrid(np.arange(x1_min-1, x1_max+1, h),np.arange(x2_min-1, x2_max+1, h))
    tree=Tree()
    tree.depth_fist_build(X, criterion='gini', min_impurity_decrease=0.00, min_samples_leaf=1, max_depth=30,threshold=0.02)
    # Z=tree.predict(np.c_[xx.ravel(), yy.ravel()],False)
    Z=tree.predict(np.c_[xx.ravel(), yy.ravel()],True)
    Z=Z.reshape(xx.shape)
    print Z
    ax = plt.subplot(1, 1, 1)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.set_title("Input data")
    ax.scatter(X.T[0], X.T[1], c=y, cmap=cm_bright,edgecolors='k')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(range(int(floor(x1_min)),int(ceil(x1_max))))
    ax.set_yticks(range(int(floor(x2_min)),int(ceil(x2_max))))
    cntr1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    cbar0 = plt.colorbar( cntr1,)
    for leaf_key in tree.leaves.keys():
        print tree.leaves[leaf_key].X,tree.leaves[leaf_key].depth, leaf_key
    plt.tight_layout()
    plt.show()










