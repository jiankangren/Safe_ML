#!/usr/bin/env python
"""
Author: Xiaozhe Gu
"""
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pudb

# FAIL_MIN_SAMPLE=-1
# FAIL_SCORE_GAIN=-2
# FAIL_IMPURITY_DECREASE=-3


def best_spliter(node,criterion,root_gain,min_samples_leaf,threshold):
	"""Find the best split index,value, and score_gain

	Parameters 
	----------
	node:  Leaf
	    Tree Leaf Node
	criterion: ScoreFunc
	    Score Function
	max_features: int  (future extension)
	    The maximum number feature to be visited
	min_samples_leaf : int
	    The minimum number of samples required to be at a leaf node:
	
	Future Extension
	----------------
	1. randomly choose feature
	
	2. randomly choose spilit value
	
	"""
	ini_score=criterion.gini_index(node.n_sample,node.n_empty)
	X=node.sortedX
	s=None
	feature_index=-1
	best_score=float('inf')
	worst_score=0
	for i in xrange(0,node.n_feature):
		cdf,bins=hist(node.sortedX[i])
		n_split=len(bins)
		if n_split<2:
			"""At least one spilit point for the current feautre"""
			break
		for j in xrange(0,n_split-1):
			s_j=bins[j]
			s_j=(bins[j]+bins[j+1])*0.5
			n_left=cdf[j]
			n_right=node.n_sample-n_left
			if n_left<min_samples_leaf or n_right<min_samples_leaf:
				"""The n_sample after spliting must satisfy the requirement"""
				continue
			score=criterion.score(node,i,s_j,n_left,n_right,node.n_empty)
			if  best_score-score>0:
				feature_index=i
				s=s_j
				best_score=score
		worst_score=max(worst_score,best_score)
	# pudb.set_trace()
	if feature_index==-1:
		print 'FAIL TO FIND ANY SPLIT POINT: minsample at depth',\
		 node.depth, 'with n_sample', node.n_sample

	if root_gain!=None and (ini_score-best_score)/root_gain<threshold:
		print 'FAIL  TO FIND ANY SPLIT POINT: not enough gain at depth',\
		 node.depth, 'with n_sample', node.n_sample
		
		feature_index=-1
	return feature_index, s,ini_score-best_score



def hist(x):
    ''' For a list of values, return the cumulative numbers
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
# 
# 	def __init__(self,index):
# 		self.index=index
# 		self.left_child=None
# 		self.right_child=None
# 		self.feature=None
# 		self.threshold=None

# 	def __repr__(self):
# 		return "{feature:%d,threshold:%d}" \
# 		% (self.left_child, self.right_child, self.feature)


		

	