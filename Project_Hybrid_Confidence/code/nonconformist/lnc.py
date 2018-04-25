#!/usr/bin/python 
# -*- coding: UTF-8 -*- 
from __future__ import division
import abc
import numpy as np
import sklearn.base






# -----------------------------------------------------------------------------
# Base Clusters
# -----------------------------------------------------------------------------

class NcCluster(sklearn.base.ClusterMixin):
	"""docstring for NcCluster"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, model,fit_params=None):
		super(NcCluster, self).__init__()
		self.model = model
		self.fit_params = {} if fit_params is None else fit_params
	def fit(X):
		self.model.fit(x, y, **self.fit_params)
	def predict(X, y=None):
		self.model.predict(x)


# -----------------------------------------------------------------------------
# Naive Separate
# -----------------------------------------------------------------------------
"""
目标： 按照一定的半径, 把input 分割。		
"""
class NaiveCluster(NcCluster):
	"""docstring for NaiveCluster"""
	def __init__(self, model,fit_params=None):
		super(NaiveCluster, self).__init__(model,fit_params=None)





# # # -----------------------------------------------------------------------------
# # Regression nonconformity scorers
# # -----------------------------------------------------------------------------
# class LNcRegressor(sklearn.base.BaseEstimator):
# 	"""Nonconformity scorer using an underlying regression model.

# 	Parameters
# 	----------
# 	model : RegressorAdapter
# 		Underlying regression model used for calculating nonconformity scores.

# 	err_func : RegressionErrFunc
# 		Error function object.

# 	normalizer : BaseScorer
# 		Normalization model.

# 	beta : float
# 		Normalization smoothing parameter. As the beta-value increases,
# 		the normalized nonconformity function approaches a non-normalized
# 		equivalent.

# 	Attributes
# 	----------
# 	model : RegressorAdapter
# 		Underlying model object.

# 	err_func : RegressionErrFunc
# 		Scorer function used to calculate nonconformity scores.

# 	See also
# 	--------
# 	ProbEstClassifierNc, NormalizedRegressorNc
# 	"""
# 	def __init__(self,
# 	             model,
# 	             err_func=AbsErrorErrFunc(),
# 	             ):
# 		super(LNcRegressor, self).__init__()
# 		self.cal_x, self.cal_y = None, None
# 		self.err_func = err_func
# 		self.model = model

# 		# If we use sklearn.base.clone (e.g., during cross-validation),
# 		# object references get jumbled, so we need to make sure that the
# 		# normalizer has a reference to the proper model adapter, if applicable.

# 	def fit(self, x, y):
# 		"""Fits the underlying model of the nonconformity scorer.

# 		Parameters
# 		----------
# 		x : numpy array of shape [n_samples, n_features]
# 			Inputs of examples for fitting the underlying model.

# 		y : numpy array of shape [n_samples]
# 			Outputs of examples for fitting the underlying model.

# 		Returns
# 		-------
# 		None
# 		"""
# 		self.model.fit(x, y)

# 	def nc_score(self, x, y=None):
# 		"""Calculates the nonconformity score of a set of samples.

# 		Parameters
# 		----------
# 		x : numpy array of shape [n_samples, n_features]
# 			Inputs of examples for which to calculate a nonconformity score.

# 		y : numpy array of shape [n_samples]
# 			Outputs of examples for which to calculate a nonconformity score.

# 		Returns
# 		-------
# 		nc : numpy array of shape [n_samples]
# 			Nonconformity scores of samples.
# 		"""
# 		prediction = self.model.predict(x)
# 		n_test = x.shape[0]
# 		return self.err_func.apply(prediction, y) 


# 	def calibrate(self, x, y):
# 		"""Calibrate conformal predictor based on underlying nonconformity
# 		scorer.

# 		Parameters
# 		----------
# 		x : numpy array of shape [n_samples, n_features]
# 			Inputs of examples for calibrating the conformal predictor.

# 		y : numpy array of shape [n_samples, n_features]
# 			Outputs of examples for calibrating the conformal predictor.

# 		increment : boolean
# 			If ``True``, performs an incremental recalibration of the conformal
# 			predictor. The supplied ``x`` and ``y`` are added to the set of
# 			previously existing calibration examples, and the conformal
# 			predictor is then calibrated on both the old and new calibration
# 			examples.

# 		Returns
# 		-------
# 		None
# 		"""
# 		self._update_calibration_set(x, y)
# 		cal_scores = self.nc_score(self.cal_x, self.cal_y)
# 		self.cal_scores = np.sort(cal_scores)[::-1]
# 	def _update_calibration_set(self, x, y):
# 		self.cal_x, self.cal_y = x, y

# 	def predict(self, x, ncset, significance=None):
# 		"""Constructs prediction intervals for a set of test examples.

# 		Predicts the output of each test pattern using the underlying model,
# 		and applies the (partial) inverse nonconformity function to each
# 		prediction, resulting in a prediction interval for each test pattern.

# 		Parameters
# 		----------
# 		x : numpy array of shape [n_samples, n_features]
# 			Inputs of patters for which to predict output values.

# 		significance : float
# 			Significance level (maximum allowed error rate) of predictions.
# 			Should be a float between 0 and 1. If ``None``, then intervals for
# 			all significance levels (0.01, 0.02, ..., 0.99) are output in a
# 			3d-matrix.

# 		Returns
# 		-------
# 		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
# 			If significance is ``None``, then p contains the interval (minimum
# 			and maximum boundaries) for each test pattern, and each significance
# 			level (0.01, 0.02, ..., 0.99). If significance is a float between
# 			0 and 1, then p contains the prediction intervals (minimum and
# 			maximum	boundaries) for the set of test patterns at the chosen
# 			significance level.
# 		"""
# 		n_test = x.shape[0]
# 		prediction = self.model.predict(x)
# 		intervals = np.zeros((x.shape[0], 2))
# 		err_dist = self.err_func.apply_inverse(nc, significance)
# 		err_dist = np.hstack([err_dist] * n_test)
# 		err_dist *= norm
# 		intervals[:, 0] = prediction - err_dist[0, :]
# 		intervals[:, 1] = prediction + err_dist[1, :]
# 		return intervals
		
if __name__=='__main__':
	from sklearn.cluster import KMeans
	from sklearn.datasets import load_boston

	boston = load_boston()
	idx = np.random.permutation(boston.target.size)

	# Divide the data into proper training set, calibration set and test set
	idx_train, idx_cal, idx_test = idx[:300], idx[300:399], idx[399:]
	X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
	kmeans = KMeans(n_clusters=5, random_state=0).fit(boston.data[idx_train, :])
	print kmeans.labels_
	print kmeans.predict(boston.data[1:3])
	print kmeans.cluster_centers_

