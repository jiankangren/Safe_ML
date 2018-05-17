import abc
from numpy import log2
import numpy as np

class ScoreFunc(object):
	"""Base class for Metric Score	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(ScoreFunc, self).__init__()

	@abc.abstractmethod
	def score(self, groups, classes): 
		"""Apply the score metric function

		Parameters
		----------
		group : numpy array of shape [[group1], [group2]...[]]
        groups: [[x1,y1],[x2,y2]....]
		classes : numpy array of shape [k_labels]
		Returns
		-------
		score : the final score
		"""
		pass

class GiniFunc(ScoreFunc):
    """
    Calculate the Gini index for a split dataset
    1-0*0-1*1< 1-0.5*0.5-0.5*0.5
    """
    def __init__(self):
        super(GiniFunc, self).__init__()
    
    @staticmethod
    def score( groups, classes):
        if not isinstance (groups,np.ndarray):
            groups=np.array(groups)
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
       # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p_index=(group[:,-1]==class_val)
                p = len(group[p_index]) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini


class InfoFunc(ScoreFunc):
    """
    Calculate the info gain ratio index for a split dataset
    """
    def __init__(self):
        super(InfoFunc, self).__init__()
    @staticmethod
    def score(groups, classes):
        if not isinstance (group,np.ndarray):
            groups=np.array(group)
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        splitcost=0
        X=[]
        for group in groups:
            X+=group
        info_origin=entropy(X,classes)
        info=0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            info+=(size/n_instances)* entropy(group,classes)
            splitcost+=-(size/n_instances)*log2(size/n_instances)
        return (info_origin-info)                
    @staticmethod
    def entropy(group,classes):
        size=float(len(group))
        info=0
        if size==0:
            return info
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if p>0:
                info+=-p*log2(p)
        return info
  


if __name__=='__main__':
    # test Gini values

    print GiniFunc.score([[[1, 1], [1, 1]], [[1, 1], [1, 0]]], [0, 1])
    # print gini.score([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1])

    # info=InfoFunc()
    # print(info.entropy([[1, 1],[1,1],[2,1], [1, 0]], [0, 1]))
    # print(info.entropy([[1, 1],[1,1], [1, 0]], [0, 1]))
    # data=np.array([[[1, 1], [1, 1],[1, 1]], [[1, 0]]])
    # print info.score(data, [0, 1])
    # print info.entropy([[1,1],[1,1],[1,1],[1,0]],[1,0])
    # print info.entropy([[1,1],[1,0]],[1,0])


