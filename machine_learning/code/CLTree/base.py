#!/usr/bin/env python
"""
Author: Xiaozhe Gu
"""
from abc import ABCMeta
from abc import abstractmethod
import numpy as np



class XData(object):
	"""
	Docstring for Training Data
	"""
	def __init__(self,X):
		"""
		Attributes
		-------------------
		X: numpy.darray 
		data = np.array([
		[-2.571244718,4.784783929,1],
		[-3.571244718,5.784783929,0],)
		    Input  data. 
		"""
		super(XData, self).__init__()
		if not isinstance (X,np.ndarray):
			raise ValueError("X should be in np.ndarray format, got %s" % type(X))
		if len(X)<=0:
			raise ValueError("Sample number should be than 0 , got %s" % len(X))
		self.X=X
		self.sortedX=np.sort(X.T)
		self.__dict__['feature_limit']=[[x[0],x[-1]] for x in self.sortedX]
		self.__dict__['n_feature']=len(X.T)
		self.__dict__['n_sample']=len(X)
	@property
	def feature_limit(self): 
		return  self.__dict__['feature_limit']
	@property
	def feature_type(self): 
		'''identify all category feature
		'''
		return None
	@property
	def n_feature(self): 
		return  self.__dict__['n_feature']
	@property
	def n_sample(self): 
		return self.__dict__['n_sample']
	def split_by_index(self,i,sl):
		#split data by feature i with value sl
		left_index=self.X.T[i]<=sl
		DL=XData(self.X[left_index])
		right_index=np.logical_not(left_index)
		DR=XData(self.X[right_index])
		return DL,DR




# class Node(object):
# 	"""
# 	Class for Tree Node
#     Attributes
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
	print GiniFunc.score( X,i=0,sl=1,sr=2,n_left=1,n_right=6,c=1)