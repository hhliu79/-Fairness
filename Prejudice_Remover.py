#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:25:48 2019

@author: huihuiliu
"""

import numpy as np

# Scalers
from sklearn.preprocessing import StandardScaler
# Bias mitigation techniques
from aif360.algorithms.inprocessing import PrejudiceRemover


np.random.seed(1)


def prejudice_remover(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups, sens_attr, ETA, arr):
    
    # Learning a Prejudice Remover (PR) model on original data
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=ETA)
    pr_orig_scaler = StandardScaler()
    dataset_orig_train.features = pr_orig_scaler.fit_transform(dataset_orig_train.features)
    pr_orig = model.fit(dataset_orig_train)
    
    # validating PR model
    thresh_arr = np.linspace(0.01, 0.50, 50)
    
    dataset_orig_valid.features = pr_orig_scaler.transform(dataset_orig_valid.features)
    dataset_orig_test.features = pr_orig_scaler.transform(dataset_orig_test.features)

    
    val_metrics = test(dataset_orig_valid, pr_orig, thresh_arr, privileged_groups, unprivileged_groups)
    pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])
    dataset_pred = test(dataset_orig_test, pr_orig, [thresh_arr[pr_orig_best_ind]], privileged_groups, unprivileged_groups)

       
    return val_metrics, dataset_pred
    
    
def test(dataset, model, thresh_arr, privileged_groups, unprivileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        
       
    return dataset_pred  
    
    
    