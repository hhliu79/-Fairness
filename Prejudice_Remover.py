#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:25:48 2019

@author: huihuiliu
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.metrics import ClassificationMetric
from collections import defaultdict
import copy

def prejudice_remover(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups, sens_attr):
    #print(dataset_orig_train.features.shape)
    #print(dataset_orig_train.labels.shape )
    dataset_original_train = copy.deepcopy(dataset_orig_train)  
    dataset_original_valid = copy.deepcopy(dataset_orig_valid)
    dataset_original_test = copy.deepcopy(dataset_orig_test)
    
    # Learning a Prejudice Remover (PR) model on original data
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
    pr_orig_scaler = StandardScaler()
    dataset_original_train.features = pr_orig_scaler.fit_transform(dataset_original_train.features)
    dataset_original_valid.features = pr_orig_scaler.fit_transform(dataset_original_valid.features)
    dataset_original_test.features = pr_orig_scaler.fit_transform(dataset_original_test.features)
    pr_orig = model.fit(dataset_original_train)
    
    # validating PR model
    thresh_arr = np.linspace(0.01, 0.50, 50)
    valid_pred, valid_metrics = test(dataset_original_valid, pr_orig, thresh_arr, privileged_groups, unprivileged_groups)
    pr_orig_best_ind = np.argmax(valid_metrics['bal_acc'])
    
    train_pred, metric_train = test(dataset_original_train, pr_orig, [thresh_arr[pr_orig_best_ind]], privileged_groups, unprivileged_groups)
    val_pred, metric_val = test(dataset_original_valid, pr_orig, [thresh_arr[pr_orig_best_ind]], privileged_groups, unprivileged_groups)
    test_pred, metric_test = test(dataset_original_test, pr_orig, [thresh_arr[pr_orig_best_ind]], privileged_groups, unprivileged_groups)
    
    dataset_transf_valid_pred = copy.deepcopy(dataset_orig_valid)
    dataset_transf_valid_pred.labels = val_pred.labels
    print(val_pred.labels.shape)
    #dataset_transf_valid_pred.scores = val_pred.scores
        
    dataset_transf_test_pred = copy.deepcopy(dataset_orig_test)
    dataset_transf_test_pred.labels = test_pred.labels
    #dataset_transf_test_pred.scores = test_pred.scores

    return dataset_transf_valid_pred, dataset_transf_test_pred
    #return val_pred, test_pred
    
    
def test(dataset, model, thresh_arr, privileged_groups, unprivileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
       
    return dataset_pred, metric_arrs  

    
    
    

'''

def prejudice_remover(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups, sens_attr, ETA):
    
    # Learning a Prejudice Remover (PR) model on original data
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=ETA)
    scale_orig = StandardScaler()
    dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
    #dataset_orig_valid.features = scale_orig.transform(dataset_orig_valid.features)
    #dataset_orig_test.features = scale_orig.transform(dataset_orig_test.features)
    model.fit(dataset_orig_train)
      
    class_thresh_arr = np.linspace(0.01, 0.50, 50)
    
    y_val_pred_prob = model.predict(dataset_orig_valid).scores
    pos_ind = 0  
    metric_arrs = defaultdict(list)
    
    for thresh in class_thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
        dataset_pred = dataset_orig_valid.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset_orig_valid, dataset_orig_valid,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
       
        
    best_ind = np.argmax(metric_arrs['bal_acc'])
    best_class_thresh = class_thresh_arr[best_ind]
    
    
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    #dataset_orig_valid_pred.scores = model.predict(dataset_orig_train)[:,pos_ind].reshape(-1,1)
    
    fav_inds1 = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds1] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds1] = dataset_orig_valid_pred.unfavorable_label    
    

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    #dataset_orig_test_pred.scores = model.predict(dataset_orig_train)[:,pos_ind].reshape(-1,1)
    
    fav_inds2 = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds2] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds2] = dataset_orig_test_pred.unfavorable_label    
    
    return dataset_orig_test_pred, dataset_orig_test_pred
'''    
