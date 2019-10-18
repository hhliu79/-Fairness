#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:05:05 2019

@author: huihuiliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:19:30 2019

@author: huihuiliu
"""

import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from warnings import warn

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from common_utils import compute_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
from collections import OrderedDict

    
def eq_odds(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups,):
    ### Train classifier on original data
    # Logistic regression classifier and predictions
    print(dataset_orig_train.features.shape)
    print(dataset_orig_train.labels.shape)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    print(X_train.shape)
    print(y_train.shape)
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred
    
    
    #### Obtain scores for validation and test sets
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
    
    
    ### Find the optimal parameters from the validation set
    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
    
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                             dataset_orig_valid_pred, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                       +classified_metric_orig_valid.true_negative_rate())
        
    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        
    #Estimate optimal parameters for the ROC method
    EO = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 seed = 1)
    EO = EO.fit(dataset_orig_valid, dataset_orig_valid_pred)
 
 
     # Metrics for the transformed test set
    dataset_transf_test_pred = EO.predict(dataset_orig_test_pred)
    
               
    return dataset_transf_test_pred
        