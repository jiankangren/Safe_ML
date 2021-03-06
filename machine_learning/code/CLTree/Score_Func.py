import abc
from numpy import log2
import numpy as np
# from ctypes import *
import numba


class ScoreFunc(object):
	"""Base class for Metric Score	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(ScoreFunc, self).__init__()

	@abc.abstractmethod
	def score(self): 
		"""
        Apply the score metric function

        Attributes
		----------
		Node :  base.Node
            The current tree node
        i: int
            Feature index
        s: float
            All left childen has ith feature  <=s
        n_left: int
            Number of samples has ith feature  <=s
        n_right  :  int 
            Number of samples has ith feature  >s
        n_empty  :  int 
            Number of empty samples has ith feature 
		-------
		score : the final score
		"""
	

class GiniFunc(ScoreFunc):
    def __init__(self):
        super(GiniFunc, self).__init__()
        # self.GINI = cdll.LoadLibrary('./GINI.so')
    def score(self,xmin,xmax,s,n_left,n_right,n_empty,min_samples_leaf):
        """
        return  Gini Score for the split : IG_gain
        """
        # xmin=node.feature_limit[0][i]
        # xmax=node.feature_limit[1][i]
        if n_left<min_samples_leaf or n_right<min_samples_leaf:
            return float('inf')
        L=float(xmax-xmin)
        E1=n_empty*((s-xmin)/L)
        E2=n_empty*((xmax-s)/L)
        # IGL=1-((n_left)/(n_left+E1))**2-((E1)/(n_left+E1))**2
        IGL=self.gini_index(n_left,E1)
        # IGR=1- ((n_right)/(n_right+E2))**2-((E2)/(n_right+E2))**2
        IGR=self.gini_index(n_right,E2)
        IG_gain=IGL*((n_left+E1)/(E1+E2+n_left+n_right))+\
            IGR*((n_right+E2)/(E1+E2+n_left+n_right))
        return  IG_gain
    def array_score(self,xmin,xmax,Ss,N_LEFT,N_RIGHT,n_empty,min_samples_leaf):
        """
        return  Gini Score for the split : IG_gain
        """
        # xmin=node.feature_limit[0][i]
        # xmax=node.feature_limit[1][i]
        def _score(s,n_left,n_right):
            return self.score(xmin,xmax,s,n_left,n_right,n_empty,min_samples_leaf)
        A_score= np.frompyfunc(_score, 3, 1)
        return A_score(Ss,N_LEFT,N_RIGHT)


    def gini_index(self,n_sample,n_empty):
        """
        return gini index 
        """
        """ origin python version
        """
        IG=1-((float(n_sample))/(n_sample+n_empty))**2-(float((n_empty))/(n_empty+n_sample))**2
        return IG

        # gini_index=self.GINI.gini_index
        # gini_index.argtype=c_float
        # gini_index.restype=c_float
        # return gini_index(c_float(n_sample),c_float(n_empty))




# class InfoFunc(ScoreFunc):
#     """
#     Calculate the info gain ratio index for a split dataset
#     """
#     def __init__(self):
#         super(InfoFunc, self).__init__()
#     @staticmethod
#     def score(groups, classes):
#         if not isinstance (group,np.ndarray):
#             groups=np.array(group)
#         # count all samples at split point
#         n_instances = float(sum([len(group) for group in groups]))
#         splitcost=0
#         X=[]
#         for group in groups:
#             X+=group
#         info_origin=entropy(X,classes)
#         info=0
#         for group in groups:
#             size = float(len(group))
#             # avoid divide by zero
#             if size == 0:
#                 continue
#             info+=(size/n_instances)* entropy(group,classes)
#             splitcost+=-(size/n_instances)*log2(size/n_instances)
#         return (info_origin-info)                
#     @staticmethod
#     def entropy(group,classes):
#         size=float(len(group))
#         info=0
#         if size==0:
#             return info
#         for class_val in classes:
#             p = [row[-1] for row in group].count(class_val) / size
#             if p>0:
#                 info+=-p*log2(p)
#         return info
  


if __name__=='__main__':
    # print GiniFunc().array_score(1,10,[2,3,4],[2,3,4],[3,2,1],10,1)
    pass


