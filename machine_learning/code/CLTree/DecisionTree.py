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
CRITERIA_CLF = {"gini":GiniFunc()}

"""
The class of tree leaf node.
    
    Parameter: 
    ----------
    feature_limit: [[min,max]*d]
        The boundary of the leaf node
    X: numpy.darray 
        data = np.array([
        [-2.571244718,4.784783929,1],
        [-3.571244718,5.784783929,0],)
        Input  data for different features.
    n_empty : float:
        Number of empty sample
    depth : int
        Depth of the node
    split_set  : set 
        The set of split feature index (0,1,2)
    
    Attribute:
    ----------
    vol : float
        Volume of the leaf node and is equal to prod(length of each dimension)
    n_sample : int
        Number of samples
    n_feature: int 
        Number of features
    density : float
        The density of the leaf node = n_sample/(n_sample+n_empty)

"""
class Leaf(object):
    """
    __slots__ = ('index','left_child', 'right_child', 'feature',
    'threshold', 'purity', 'n_node_samples','score')
    """

    def __init__(self,X,n_empty=0,depth=1,feature_limit=None,\
            split_set_L=set([]),split_set_R=set([])):
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
        self.split_set_L=split_set_L
        self.split_set_R=split_set_R
    @property
    def feature_limit_X(self):
        S=[[f[0],f[1]] for f in self.feature_limit]
        full_set=set(range(0,len(self.X.T)))
        for i in full_set^self.split_set_L:
            S[i][0]=self.sortedX[i][0] 
        for i in full_set^self.split_set_R:
            S[i][1]=self.sortedX[i][-1] 
        return S
    

    @property
    def vol(self): 
        return np.prod(np.array([feature_limit[1]-feature_limit[0] \
            for feature_limit in self.feature_limit]))
    @property
    def vol_X(self): 
        return np.prod(np.array([feature_limit[1]-feature_limit[0] \
            for feature_limit in self.feature_limit_X]))
    @property
    def feature_gap(self):
        return None
    @property
    def n_sample(self): 
        return self.__dict__['n_sample']
    @property
    def n_feature(self): 
        return self.__dict__['n_feature']
    
    """
    Split data by feature i with value s, and return node_left,node_right
    """
    def split_by_index(self,i,s,n_empty_left,n_empty_right):

        left_index=self.X.T[i]<=s
        s1=set(self.split_set_L)
        s2=set(self.split_set_R)
        s2.add(i)
        node_left=Leaf(self.X[left_index],n_empty_left,self.depth+1,\
            self.feature_limit,s1,s2)
        node_left.feature_limit[i][1]=s


        right_index=np.logical_not(left_index)
        s1=set(self.split_set_L)
        s1.add(i)
        s2=set(self.split_set_R)
        node_right=Leaf(self.X[right_index],n_empty_right,self.depth+1,\
            self.feature_limit,s1,s2)
        node_right.feature_limit[i][0]=s

        
        return   node_left,   node_right

"""
The Clustering Tree
    Attribute
    ----------
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
    leaves : dict
        Contains leaf nodes 
    vol : float
        Volume of the tree  = prod(x_imax-x_imin for i)
    total_sample: int
        The number of total training samples
    root_gain: float 
        The initial score (1-p0^2-p1^2)  (- /minus ) the best score for the first split
    density {ID: density}
    density_X {ID: density}
  
    Examples
         0
       1    2
      3 4  5 6
    7
    feature :   np.array([0,1,1,2,1,0,1])
    children_left  [1,3,5,7,-1,-1,-1,-1]
    children_right  [2,4,6,-1,-1,-1,-1,-1]
    threshold  [0.5,0.19,1,3...]

    Parameter:
    ----------
    X : np.darray
        The input data
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node:
    max_depth : int 
        maximum depth of the tree to stop
    gain_ratio_threshold: float
        If (ini_score-best_score)/root_gain < threshold , then split
"""


class CLTree(object):
   
    def __init__(self, criterion='gini', min_samples_leaf=1, max_depth=10,gain_ratio_threshold=0.01):
        
        super(CLTree, self).__init__()
        self.children_left=np.zeros(CONSTANT_SIZE,dtype=int)
        self.children_right=np.zeros(CONSTANT_SIZE,dtype=int)
        self.feature=np.zeros(CONSTANT_SIZE,dtype=int)
        self.threshold_left=np.zeros(CONSTANT_SIZE)
        self.threshold_right=np.zeros(CONSTANT_SIZE)
        self.density=np.zeros(CONSTANT_SIZE)
        self.density_X=np.zeros(CONSTANT_SIZE)
        self.vol=0
        self.total_sample=0
        self.root_gain=None
        self.leaves={}


        self.criterion = CRITERIA_CLF[criterion]
        self.min_samples_leaf=min_samples_leaf
        self.max_depth=max_depth
        self.gain_ratio_threshold=gain_ratio_threshold
     
    
    """Given percentage p=0.3, give the corresponding density value
    """
    def ICDF_threshold(self,p):
       
        if type(p) is float:
            S=self.density_X[self.density_X>0]
            S.sort()
            i=int(p*len(S))
            return S[i]
        elif type(p) is list: 
            S=self.density_X[self.density_X>0]
            S.sort()
            for i in xrange(0,len(p)):
                k=int(p[i]*len(S))
                p[i]=S[k]
            return p


    """Allocate more memory to array attributes
    """
    def memory_update(self,size=100):
        self.feature=np.append(self.feature,np.zeros(size,dtype=int))
        self.children_left=np.append(self.children_left,np.zeros(size,dtype=int))
        self.children_right=np.append(self.children_right,np.zeros(size,dtype=int))
        self.threshold_left=np.append(self.threshold_left,np.zeros(size))
        self.threshold_right=np.append(self.threshold_right,np.zeros(size))
        self.density=np.append(self.density,np.zeros(size))
        self.density_X=np.append(self.density_X,np.zeros(size))
    
    """ Determine the leaf ID of for X
        
        Parameter:
        ----------
        X :  [np.darray * n_predict_sample]
            Input Data
        
        Return:
        -------
        Y : [int * n_predict_sample]
            The leaf node ID each input belong to
    """
    def decision_func(self,X):
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

    """ Predict the extent the input data belong to confident region,  and 
        is equal to 1 if the corresponding density >threshold

        Parameter:
        ----------
        X :  [np.darray * n_predict_sample]
            Input Data
        threshold : float
            If the predict value > threshold, then x belong to the confident
            region.
        
        Return:
        -------
        Y : [float * n_predict_sample]
            0 denotes empty space and 1 denotes confident region
    """
    def predict(self,X,threshold=0.5):
        Y=self.decision_func(X)
        for i in xrange(0,len(Y)):
            if Y[i]==-1:
                Y[i]=0
            else :
                ID=int(Y[i])
                x=X[i]
                if_satisfy=np.prod([int(self.leaves[ID].feature_limit_X[j][0]<=x[j]\
                    <= self.leaves[ID].feature_limit_X[j][1]) for j in xrange(0,len(x))])>0
                if if_satisfy:
                    Y[i]= self.density_X[ID]
                elif np.prod([int(self.leaves[ID].feature_limit[j][0]<=x[j]\
                    <= self.leaves[ID].feature_limit[j][1]) for j in xrange(0,len(x))])>0:
                    Y[i]=self.density[ID]
                else:
                    Y[i]=0
        # Y=(Y>threshold)+np.zeros(len(Y))
        return np.array(Y)







    """ Build the tree 
    """
    def fit(self,X):

        if   self.max_depth<=1:
            raise ValueError("Tree depth should be greater than 1, get  %d" %(max_depth))
        n_feature=len(X.T)
        heap=[0] # heap for construct the tree
        i=0 # keep node index
        root=Leaf(X,n_empty=len(X),depth=1)
        self.leaves[0]=root

        '''total_sample, total volumne'''
        self.vol=root.vol
        self.total_sample=root.n_sample

        if  root.n_sample< self.min_samples_leaf:
                raise ValueError("Root should has more than %d samples, get  %d" \
                    %(min_samples_leaf,root.n_sample))
        elif  root.n_sample== self.min_samples_leaf:
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

            index,s,score_gain=best_spliter(current_node,self.criterion,self.root_gain,\
             self.min_samples_leaf, self.gain_ratio_threshold)
            
            if current_node.depth==1:
                self.root_gain=score_gain
            if index==FAIL_TO_FIND:
                '''If fail to find any split, then the current node is a leaf
                '''
                self.feature[current_node_index]=IS_LEAF
                self.leaves[current_node_index]=current_node  

                self.density[current_node_index] =   current_node.n_sample/ \
                    (current_node.n_sample+self.total_sample * current_node.vol/self.vol)              
                self.density_X[current_node_index]= current_node.n_sample/ \
                    (current_node.n_sample + self.total_sample * current_node.vol_X/self.vol) 
            

                continue

            xmin=current_node.feature_limit[index][0]
            xmax=current_node.feature_limit[index][1]
            L=float(xmax-xmin)

            
            n_empty_left=current_node.n_empty*((s-xmin)/L)
            n_empty_right=current_node.n_empty*((xmax-s)/L)

            node_left,node_right=current_node.split_by_index(index,s,n_empty_left,n_empty_right)

            self.feature[current_node_index]=index
            self.threshold_left[current_node_index]= node_left.feature_limit[index][1]
            self.threshold_right[current_node_index] = node_right.feature_limit[index][0]

            

            i+=1
            id_left=i
            self.children_left[current_node_index]=id_left
            if   node_left.depth>= self.max_depth or node_left.n_sample<= self.min_samples_leaf:                
                self.feature[id_left]=IS_LEAF
                self.density[id_left] =     node_left.n_sample/ \
                    (node_left.n_sample+self.total_sample * node_left.vol/self.vol)              
                self.density_X[id_left]=    node_left.n_sample/ \
                    (node_left.n_sample + self.total_sample * node_left.vol_X/self.vol)
              
            else:
                heap.append(id_left)
            self.leaves[id_left]=node_left
            
            i+=1
            id_right=i
            self.children_right[current_node_index]=id_right
            if   node_right.depth>= self.max_depth or node_right.n_sample<= self.min_samples_leaf:                
                self.feature[id_right]=IS_LEAF
                self.density[id_right] =   node_right.n_sample/ \
                    (node_right.n_sample+ self.total_sample  * node_right.vol/self.vol)              
                self.density_X[id_right]=  node_right.n_sample/ \
                    (node_right.n_sample +self.total_sample * node_right.vol_X/self.vol)
             

            else:
                heap.append(id_right)
            self.leaves[id_right]=node_right



       







if __name__=='__main__':
    n=200
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
    print X
    # x11=np.linspace(0, 4, num=250)
    # x12=np.linspace(6, 10, num=250)
    # x1=np.concatenate((x11, x12), axis=0)
    # x2=map(lambda x:np.sin(x)+np.random.normal(0,0.1),x1 )
    # X=np.array([x1,x2]).T
    # y=np.ones(len(X))

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
    y=np.ones(len(X))
    
    # X= np.array([[0,0],[1,1,],[3,3],[4,4],[6,6],[7,7],[9,9],[10,10]])
    # y=np.ones(len(X))

    h=0.05
    x1_min, x1_max = X.T[0].min() - .01, X.T[0].max() + .01
    x2_min, x2_max = X.T[1].min() - .01, X.T[1].max() + .01
    xx, yy = np.meshgrid(np.arange(x1_min-1, x1_max+1, h),np.arange(x2_min-1, x2_max+1, h))
    tree=CLTree(criterion='gini',min_samples_leaf=2, max_depth=50,gain_ratio_threshold=0.1)
    tree.fit(X)
    Z=tree.predict(np.c_[xx.ravel(), yy.ravel()],threshold=0.45)
    Z=Z.reshape(xx.shape)
    # print Z
    ax = plt.subplot(1, 1, 1)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.set_title("")
    ax.scatter(X.T[0], X.T[1], c=y, cmap=cm_bright,edgecolors='k')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(range(int(floor(x1_min)),int(ceil(x1_max))))
    ax.set_yticks(range(int(floor(x2_min)),int(ceil(x2_max))))
    cntr1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    cbar0 = plt.colorbar( cntr1,)
    # print tree.feature
    # print tree.threshold_left
    # print tree.children_left
    # print tree.threshold_right
    # print tree.children_right
    # for leaf_key in tree.leaves.keys():
    #     print '----'
    #     print tree.leaves[leaf_key].X
    #     print '~~~~'
    #     print tree.leaves[leaf_key].feature_limit_X
    # print tree.ICDF_threshold([0.2,0.4,0.6,0.8])
    plt.tight_layout()
    plt.show()











