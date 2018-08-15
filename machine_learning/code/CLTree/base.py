#!/usr/bin/env python
"""
Author: Xiaozhe Gu
"""
from abc import ABCMeta
from abc import abstractmethod
import numpy as np



def best_spliter(node,criterion,min_samples_leaf=1):
	"""
	Attributes
	----------
	node:  Node
	    Tree Node
	criterion: ScoreFunc
	    Score Function
	max_features: int
	    The maximum number feature to be visited
	min_samples_leaf : int
	    The minimum number of samples required to be at a leaf node:
	
	Future Extension
	----------------
	1. randomly choose feature
	
	2. randomly choose spilit value
	
	"""
	X=node.sortedX
	s=None
	feature_index=-1
	best_score=float('inf')
	for i in xrange(0,node.n_feature):
		cdf,bins=hist(node.sortedX[i])
		n_split=len(bins)
		for j in xrange(0,n_split):
			s_j=bins[j]
			n_left=cdf[j]
			n_right=node.n_sample-n_left
			if n_left<min_samples_leaf or n_right<min_samples_leaf:
				continue
			score=criterion.score(node,i,s_j,n_left,n_right,node.n_empty)
			if score<best_score:
				feature_index=i
				s=s_j
				best_score=score
	return feature_index, s



def hist(x):
    '''
    Parameter:
    ----------
    x: np.darray
        A list sorted of feature values  x=[1, 1,1,2,2,3,4,4]
    Return:
    ------
    cdf: np.darray
        A list of cumulative features     array([3, 5, 6, 8]) 
        x has 3 items <=1, 5 items <=2, 6 items <=3, 8 items <=4
    bins:  np.darray                        
        A list of s                       array([1, 2, 3, 4])
    '''
    feature_set=list(set(x))
    feature_set.sort()
    # print 'feature set:', feature_set
    bins=[x[0]-1]+list(feature_set)+[x[-1]+1]
    hist,bins=np.histogram(x,bins)
    cdf=np.cumsum(hist)
    return cdf[1:], bins[1:-1]




# class Node(object):
# 	"""
# 	Class for Tree Node
#   Attributes
# 	-------------------
# 	left_child :  node
# 		The left child node of the current node
# 	right_child : node
# 		The right child node of the current node
# 	feature: int
# 		Split index for the data
# 	sl : float 
# 		All left childen has feature  >=sl
# 	sr : float
# 		All right childen has feature >=sr
# 	n_node_samples : int
# 		Number of training samples reaching  the current node
# 	score : float
# 		The score from the split 
# 	purity : float
# 		The purity of empty points
#     """
# 	__slots__ = ('index','left_child', 'right_child', 'feature',
# 	 'threshold', 'purity', 'n_node_samples','score')
# 	def __init__(self,index):
# 		self.index=index
# 		self.left_child=None
# 		self.right_child=None
# 		self.feature=None
# 		self.threshold=None

# 	def __repr__(self):
# 		return "{feature:%d,threshold:%d}" \
# 		% (self.left_child, self.right_child, self.feature)


		

	