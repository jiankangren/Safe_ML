#!/usr/bin/env python

"""
Inductive conformal predictors.
"""


from __future__ import division
from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator


from nonconformist.util import calc_p


