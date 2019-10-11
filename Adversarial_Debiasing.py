#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:54:30 2019

@author: huihuiliu
"""

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from aif360.metrics import ClassificationMetric
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import numpy as np
from collections import defaultdict
import copy


def adversarial_debiasing(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups):
    
    dataset_original_train = copy.deepcopy(dataset_orig_train)
    dataset_original_valid = copy.deepcopy(dataset_orig_valid)
    dataset_original_test = copy.deepcopy(dataset_orig_test)
    
    tf.reset_default_graph()
    sess = tf.Session()
    debised_classifier = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debised_classifier',
                          debias=True,
                          sess=sess)
    debised_classifier.fit(dataset_original_train)
    
    dataset_nodebiasing_valid = debised_classifier.predict(dataset_original_valid)
    dataset_nodebiasing_test = debised_classifier.predict(dataset_original_test)
    sess.close()
       
    return dataset_nodebiasing_valid, dataset_nodebiasing_test



'''
def adversarial_debiasing(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups):
    
    pr_orig_scaler = StandardScaler()
    dataset_orig_train.features = pr_orig_scaler.fit_transform(dataset_orig_train.features)
    
    tf.reset_default_graph()
    sess = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='plain_classifier',
                          debias=True,
                          sess=sess)
    plain_model.fit(dataset_orig_train)
    #dataset_nodebiasing_valid = plain_model.predict(dataset_orig_valid)
    #dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)
    #print(dataset_nodebiasing_test)
    #return dataset_nodebiasing_valid, dataset_nodebiasing_test
    
    
    # validating PR model
    thresh_arr = np.linspace(0.01, 0.50, 50)
    
    valid_pred, valid_metrics = test(dataset_orig_valid, plain_model, thresh_arr, privileged_groups, unprivileged_groups)
    pr_orig_best_ind = np.argmax(valid_metrics['bal_acc'])
    val_pred, metric_val = test(dataset_orig_test, plain_model, [thresh_arr[pr_orig_best_ind]], privileged_groups, unprivileged_groups)
    test_pred, metric_test = test(dataset_orig_test, plain_model, [thresh_arr[pr_orig_best_ind]], privileged_groups, unprivileged_groups)
    
    dataset_transf_valid = copy.deepcopy(dataset_orig_valid)
    dataset_transf_valid.labels = val_pred.labels
    dataset_transf_valid.scores = val_pred.scores
    
    
    dataset_transf_test = copy.deepcopy(dataset_orig_test)
    dataset_transf_test.labels = test_pred.labels
    dataset_transf_test.scores = test_pred.scores

    sess.close()
    #return val_pred, test_pred
    return dataset_transf_valid, dataset_transf_test

    
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

