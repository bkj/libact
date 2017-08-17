#!/usr/bin/env python

"""

    linear_svc.py

    !! Only supports two classes

"""

import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from sklearn import svm

from libact.base.interfaces import ContinuousModel

class LinearSVC(ContinuousModel):
    def __init__(self, *args, **kwargs):
        self.model = svm.LinearSVC(*args, **kwargs)
        
    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
        
    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)
        
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        return np.vstack((-dvalue, dvalue)).T
