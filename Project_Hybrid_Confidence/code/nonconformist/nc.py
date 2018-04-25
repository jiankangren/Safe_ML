#!/usr/bin/env python

"""
Nonconformity Regressor
"""

# Authors: 

from __future__ import division

import abc
import numpy as np
import sklearn.base
from   base import *
# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------

class RegressionErrFunc(object):
	"""Base class for regression model error functions.
	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(RegressionErrFunc, self).__init__()

	@abc.abstractmethod
	def apply(self, prediction, y):#, norm=None, beta=0):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass

	@abc.abstractmethod
	def apply_inverse(self, nc, significance):#, norm=None, beta=0):
		"""Apply the inverse of the nonconformity function (i.e.,
		calculate prediction interval).

		Parameters
		----------
		nc : numpy array of shape [n_calibration_samples]
			Nonconformity scores obtained for conformal predictor.

		significance : float
			Significance level (0, 1).

		Returns
		-------
		interval : numpy array of shape [n_samples, 2]
			Minimum and maximum interval boundaries for each prediction.
		"""
		pass



class AbsErrorErrFunc(RegressionErrFunc):
	"""Calculates absolute error nonconformity for regression problems.

		For each correct output in ``y``, nonconformity is defined as

		.. math::
			| y_i - \hat{y}_i |
	"""
	def __init__(self):
		super(AbsErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return np.abs(prediction - y)

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		border = int(np.floor(significance * (nc.size + 1))) - 1
		# TODO: should probably warn against too few calibration examples
		border = min(max(border, 0), nc.size - 1)
		return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):
	"""Calculates signed error nonconformity for regression problems.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		y_i - \hat{y}_i

	References
	----------
	.. [1] Linusson, Henrik, Ulf Johansson, and Tuve Lofstrom.
		Signed-error conformal regression. Pacific-Asia Conference on Knowledge
		Discovery and Data Mining. Springer International Publishing, 2014.
	"""

	def __init__(self):
		super(SignErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return (prediction - y)

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		upper = int(np.floor((significance / 2) * (nc.size + 1)))
		lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
		# TODO: should probably warn against too few calibration examples
		upper = min(max(upper, 0), nc.size - 1)
		lower = max(min(lower, nc.size - 1), 0)
		return np.vstack([-nc[lower], nc[upper]])


# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------




class Nc_Reg_Creator(object):
	@staticmethod
	def create_nc(model, err_func=None, normalizer_model=None):
		err_func = AbsErrorErrFunc() if err_func is None else err_func
		regressor = RegressorAdapter(model)
		return NcRegressor(regressor, err_func)



# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class NcRegressor(sklearn.base.BaseEstimator):
	"""Nonconformity scorer using an underlying regression model.

	Parameters
	----------
	model : RegressorAdapter
		Underlying regression model used for calculating nonconformity scores.

	err_func : RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.

	Attributes
	----------
	model : RegressorAdapter
		Underlying model object.

	err_func : RegressionErrFunc
		Scorer function used to calculate nonconformity scores.

	See also
	--------
	ProbEstClassifierNc, NormalizedRegressorNc
	"""
	def __init__(self,
	             model,
	             err_func=AbsErrorErrFunc()):
		super(NcRegressor, self).__init__()
		self.cal_x, self.cal_y = None, None
		self.err_func = err_func
		self.model = model

		# If we use sklearn.base.clone (e.g., during cross-validation),
		# object references get jumbled, so we need to make sure that the
		# normalizer has a reference to the proper model adapter, if applicable.

	def fit(self, x, y):
		"""Fits the underlying model of the nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the underlying model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the underlying model.

		Returns
		-------
		None
		"""
		self.model.fit(x, y)

	def nc_score(self, x, y=None):
		"""Calculates the nonconformity score of a set of samples.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for which to calculate a nonconformity score.

		y : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of samples.
		"""
		prediction = self.model.predict(x)
		n_test = x.shape[0]
		return self.err_func.apply(prediction, y) 


	def calibrate(self, x, y):
		"""Calibrate conformal predictor based on underlying nonconformity
		scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for calibrating the conformal predictor.

		y : numpy array of shape [n_samples, n_features]
			Outputs of examples for calibrating the conformal predictor.

		increment : boolean
			If ``True``, performs an incremental recalibration of the conformal
			predictor. The supplied ``x`` and ``y`` are added to the set of
			previously existing calibration examples, and the conformal
			predictor is then calibrated on both the old and new calibration
			examples.

		Returns
		-------
		None
		"""
		self._update_calibration_set(x, y)
		cal_scores = self.nc_score(self.cal_x, self.cal_y)
		self.cal_scores = np.sort(cal_scores)[::-1]
	def _update_calibration_set(self, x, y):
		self.cal_x, self.cal_y = x, y

	def predict(self, x, nc, significance=None):
		"""Constructs prediction intervals for a set of test examples.

		Predicts the output of each test pattern using the underlying model,
		and applies the (partial) inverse nonconformity function to each
		prediction, resulting in a prediction interval for each test pattern.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then intervals for
			all significance levels (0.01, 0.02, ..., 0.99) are output in a
			3d-matrix.

		Returns
		-------
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		n_test = x.shape[0]
		prediction = self.model.predict(x)
		norm = np.ones(n_test)
		intervals = np.zeros((x.shape[0], 2))
		err_dist = self.err_func.apply_inverse(nc, significance)
		err_dist = np.hstack([err_dist] * n_test)
		err_dist *= norm
		intervals[:, 0] = prediction - err_dist[0, :]
		intervals[:, 1] = prediction + err_dist[1, :]
		return intervals


if __name__=='__main__':
	# print func
	# SIN=OptModel(None)
	# SIN.set_func(func)
	# print SIN.predict([[1],[2],[3]])
	X=np.linspace(0, 10, num=10000)
	Y1=map(lambda x:np.sin(x)+np.random.normal(0,0.1),X[0:3000])
	Y2=map(lambda x:np.sin(x)+np.random.normal(0,0.3),X[3000:5000])
	Y3=map(lambda x:np.sin(x)+np.random.normal(0,0.1),X[5000:])
	Y=np.array(Y1+Y2+Y3)
	idx = np.random.permutation(10000)
	idx_train, idx_cal, idx_test = idx[:3000], idx[3000:3999], idx[3999:]
	
	func=np.sin
	model=OptModel(None)
	model.set_func(np.sin)
	# model = RandomForestRegressor()
	nc_regressor = Nc_Reg_Creator.create_nc(model)
	# nc_regressor.fit(X.reshape(-1 ,1)[idx_train], Y[idx_train])
	
	nc_regressor.calibrate(X.reshape(-1 ,1)[idx_cal], Y[idx_cal])
	nc_set=nc_regressor.cal_scores
	print nc_set.shape
	prediction =nc_regressor.predict(X.reshape(-1 ,1)[idx_test],nc_set,significance=0.02)
	# 	