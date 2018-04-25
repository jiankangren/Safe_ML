#!/usr/bin/env python

"""
docstring
"""

# Authors: Gu Xiaozhe

import abc
import numpy as np

from sklearn.base import BaseEstimator




class BaseModel(BaseEstimator):
	__metaclass__ = abc.ABCMeta

	def __init__(self, model, fit_params=None):
		"""
		Parameters
		----------
		model: base regression model like LR
		fit_params:

		"""
		super(BaseModel, self).__init__()
		self.model = model
		self.fit_params = {} if fit_params is None else fit_params

	def fit(self, x, y):
		"""Fits the model.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the model.

		Returns
		-------
		None
		"""

		self.model.fit(x, y, **self.fit_params)
	def predict(self, x):
		"""Returns the prediction made by the underlying model.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of test examples.

		Returns
		-------
		y : numpy array of shape [n_samples]
			Predicted outputs of test examples.
		"""
		self.last_y = self._underlying_predict(x)
		return self.last_y.copy()

	@abc.abstractmethod
	def _underlying_predict(self, x):
		"""Produces a prediction using the encapsulated model.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of test examples.

		Returns
		-------
		y : numpy array of shape [n_samples]
			Predicted outputs of test examples.
		"""
		pass




class RegressorAdapter(BaseModel):
	def __init__(self, model, fit_params=None):
		super(RegressorAdapter, self).__init__(model, fit_params)

	def _underlying_predict(self, x):
		return self.model.predict(x)



class OptModel(BaseModel):
	"""docstring for OriginModel"""
	def __init__(self, model, fit_params=None):
		super(OptModel, self).__init__(model, fit_params)
	def set_func(self,func):
		self.func=func
	def fit(self, x, y):
		#do nothing
		pass
	def _underlying_predict(self, x):
		prediction=self.func(x)
		size=prediction.size
		prediction=prediction.reshape(size)
		return  prediction
		

	

