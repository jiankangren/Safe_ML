#!/usr/bin/env python
"""
Author: Xiaozhe Gu
"""
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pudb
from numpy import frompyfunc
import numba
import random
from math import floor


"""Find the best split index,value, and score_gain

Parameters 
----------
node:  Leaf
	Tree Leaf Node
criterion: ScoreFunc
	Score Function
max_feature: int  (future extension)
	The maximum number feature to be visited
min_samples_leaf : int
	The minimum number of samples required to be at a leaf node:

Future Extension
----------------
1. randomly choose feature

2. randomly choose spilit value

"""

def best_spliter(node,criterion,root_gain,min_samples_leaf,threshold,\
		max_feature_p,max_try_p):
	ini_score=criterion.gini_index(node.n_sample,node.n_empty)
	X=node.sortedX
	s=None
	feature_index=-1
	max_feature=int(floor(max_feature_p*node.n_feature))
	tmp=range(0,node.n_feature)
	try_features=random.sample(tmp,max_feature)
	
	"""BEST_SCORE: store score for index
	 BEST_S : store best split value split value """
	BEST_SCORE=[float('inf')]*node.n_feature
	BEST_S=[0]*node.n_feature
	"""[feature i:(cdf: array([1, 3, 4]), bins: array([1, 2, 3]))]
	"""
	R=map(hist,X)	

	for i in try_features:
		cdf,bins=R[i][0],R[i][1]
		n_split=len(bins)
		if n_split<2:
			"""At least one spilit point for the current feautre"""
			break
		max_try=int(max(1,floor((len(bins)-1)*max_try_p)))
		tmp=range(0,len(bins)-1)
		try_split_index=random.sample(tmp,max_try)
		xmin=node.feature_limit[0][i]
		xmax=node.feature_limit[1][i]
		Ss=np.array([(bins[j]+bins[j+1])*0.5 for j in xrange(0,len(bins)-1)])
		Ss=Ss[try_split_index]

		N_LEFT=cdf[try_split_index]
		N_RIGHT=np.array([node.n_sample-n_left for n_left in N_LEFT])
		scores=criterion.array_score(xmin,xmax,Ss,N_LEFT,N_RIGHT,node.n_empty,min_samples_leaf)
		best_score_index= np.argmin(scores)
		BEST_SCORE[i]=scores[best_score_index]
		BEST_S[i]=Ss[best_score_index]
	
	feature_index=np.argmin(BEST_SCORE)
	best_score=BEST_SCORE[feature_index]
	s=BEST_S[feature_index]
	if best_score==float('inf'):
		feature_index=-1
	if feature_index==-1:
		if __name__=='__main__':
			print 'FAIL TO FIND ANY SPLIT POINT: minsample at depth',\
			node.depth, 'with n_sample', node.n_sample
		pass

	if root_gain!=None and (ini_score-best_score)/root_gain<threshold:
		if __name__=='__main__':
			print 'FAIL  TO FIND ANY SPLIT POINT: not enough gain at depth',\
					node.depth, 'with n_sample', node.n_sample
		feature_index=-1
	return feature_index, s,ini_score-best_score



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
# @numba.autojit
def hist(x):
	feature_set=list(set(x))
	feature_set.sort()
	# print 'feature set:', feature_set
	bins=[x[0]-1]+feature_set+[x[-1]+1]
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


		

	