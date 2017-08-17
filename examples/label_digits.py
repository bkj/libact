#!/usr/bin/env python

import copy
import numpy as np

from rsub import *
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import InteractiveLabeler


# --
# Params

n_classes = 5
n_labeled = 5

# --
# IO

digits = load_digits(n_class=n_classes)  # consider binary case
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
while len(np.unique(y_train[:n_labeled])) < n_classes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

train  = Dataset(X_train, np.hstack([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
train2 = copy.deepcopy(train)
test   = Dataset(X_test, y_test)

# --
# Training

ures, rres = [], []

model = LogisticRegression()
unom = UncertaintySampling(train, method='lc', model=LogisticRegression())
rnom = RandomSampling(train2)

_ = model.train(train)
ures = np.append(ures, 1 - model.score(test))

_ = model.train(train2)
rres = np.append(rres, 1 - model.score(test))

n_labels = 250
ures, rres = np.zeros(n_labels), np.zeros(n_labels)
for i in range(n_labels):
    if not i % 10:
        print 'n_labeled=%d' % i
    
    # Uncertainty sampling
    ask_id = unom.make_query()
    train.update(ask_id, y_train[ask_id])
    _ = model.train(train)
    ures[i] = model.score(test)
    
    # Random search
    ask_id = rnom.make_query()
    train2.update(ask_id, y_train[ask_id])
    _ = model.train(train2)
    rres[i] = model.score(test)


_ = plt.plot(ures)
_ = plt.plot(rres)
show_plot()

np.where(ures > 0.95)[0][0]
np.where(rres > 0.95)[0][0]
