#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:07:32 2019

@author: huihuiliu
"""

# Load all necessary packages
import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from warnings import warn

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification
from common_utils import compute_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
from collections import OrderedDict

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05

def reject_option(dataset_orig_valid, dataset_orig_valid_pred, dataset_orig_test,dataset_orig_test_pred, privileged_groups, unprivileged_groups):
    
       
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
    
    
    
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name="Statistical parity difference",
                                  metric_ub=metric_ub, metric_lb=metric_lb)
    
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
    
    
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
  
    return dataset_transf_test_pred
    '''
    
    
    
    dataset_orig_valid_pred.labels = np.expand_dims(dataset_orig_valid_pred.labels, axis=1)
    dataset_orig_test_pred.labels = np.expand_dims(dataset_orig_test_pred.labels, axis=1)

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
    
    
    #Estimate optimal parameters for the ROC method
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name="Statistical parity difference",
                                  metric_ub=metric_ub, metric_lb=metric_lb)
    
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
   
    # Transform the validation set
    #dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)
    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    
  
    return dataset_transf_test_pred
    '''
        
