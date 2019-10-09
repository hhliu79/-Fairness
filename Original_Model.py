#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:28:33 2019

@author: huihuiliu
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
from aif360.metrics import ClassificationMetric
from tqdm import tqdm
from IPython.display import Markdown, display
from common_utils import compute_metrics
from sklearn.preprocessing import StandardScaler


def train(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups):
    
    ### Train classifier on original data
    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    lmod = LogisticRegression(solver='lbfgs', multi_class='auto')
    lmod.fit(X_train, y_train, sample_weight = dataset_orig_train.instance_weights)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred
    

    # Obtain scores for validation and test sets
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
    
    
    # Find the optimal parameters from the validation set
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

   
    return dataset_orig_valid_pred, dataset_orig_test_pred
    

