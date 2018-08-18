import abc
from numpy import log2
import numpy as np



class ScoreFunc(object):
	"""Base class for Metric Score	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(ScoreFunc, self).__init__()

	@abc.abstractmethod
	def score(self, Node,i,s,n_left,n_right,n_empty): 
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
		pass

class GiniFunc(ScoreFunc):
    def __init__(self):
        super(GiniFunc, self).__init__()
    
    @staticmethod
    def score(node,i,s,n_left,n_right,n_empty):
        """
        return  Gini Score for the split : IG_gain
        """
        xmin=node.feature_limit[i][0]
        xmax=node.feature_limit[i][1]
        L=float(xmax-xmin)
        E1=n_empty*((s-xmin)/L)
        E2=n_empty*((xmax-s)/L)
        # IGL=1-((n_left)/(n_left+E1))**2-((E1)/(n_left+E1))**2
        IGL=GiniFunc.gini_index(n_left,E1)
        # IGR=1- ((n_right)/(n_right+E2))**2-((E2)/(n_right+E2))**2
        IGR=GiniFunc.gini_index(n_right,E2)
        IG_gain=IGL*((n_left+E1)/(E1+E2+n_left+n_right))+\
            IGR*((n_right+E2)/(E1+E2+n_left+n_right))
        return  IG_gain
    @staticmethod
    def gini_index(n_sample,n_empty):
        """
        return gini index 
        """
        IG=1-((float(n_sample))/(n_sample+n_empty))**2-(float((n_empty))/(n_empty+n_sample))**2
        return  IG



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
    pass




