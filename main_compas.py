#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:13:18 2019

@author: huihuiliu
"""

# Load all necessary packages
import sys
import copy
import numpy as np
sys.path.append("../")
from aif360.metrics import BinaryLabelDatasetMetric
from loaddata import LoadData
from Original_Model import train
from pre import Pre
import numpy as np
from Adversarial_Debiasing import adversarial_debiasing
from ART_Classifier import art_classifier
from Prejudice_Remover import prejudice_remover
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.metrics import ClassificationMetric
from common_utils import compute_metrics
from collections import OrderedDict
from Calibrated_Eqodd import calibrated_eqodd
from Eq_Odds import eq_odds
from Reject_Option import reject_option
import os
#os.environ['CUDA_VISIBLE_DEVICES']='7'

#dataset = ["adult", "compas", "german", "bank"]  
dataset = ["adult", "german", "compas"]  
attribute = ["sex", "race", "age"]
preAlgorithm = ["disparate_impact_remover", "lfr", "optim", "reweighing"] 
inAlgorithm = ["adversarial_debiasing", "art_classifier", "prejudice_remover"]
#inAlgorithm = ["adversarial_debiasing", "art_classifier"]
postAlgorithm = ["calibrated_eqodds", "eqodds", "reject_option"]    

# Verify metric name
allowed_metrics = ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]    
cost_constraint = "fnr" # "fnr", "fpr", "weighted"
metric_ub = 0.05
metric_lb = -0.05
       

def main():
    '''
    dataset_used = "compas"
    for i in range(2):
        if i == 0:
            arr = dataset_used + " " + "[1, 0, 0]"
        if i == 1:
            arr = dataset_used + " " + "[0, 1, 0]"
        protected_attribute_used = attribute[i]
        for l in range(5):
            for m in range(4):
                for n in range(4):
                    dataset_orig, privileged_groups, unprivileged_groups, optim_options = LoadData(dataset_used, protected_attribute_used)                
                    algorithm_used = arr + " " + "[" + " " + str(l) + " " + str(m) + " " + str(n) + "]"
                    #dataset_original = copy.deepcopy(dataset_orig)
                    feature = comb_algorithm(l, m, n, dataset_orig, privileged_groups, unprivileged_groups, optim_options)
                    result = str(algorithm_used + feature)
                    print(result)
                    with open('comb_algor_compas544.txt', 'a') as f:
                        f.write(result)
                        f.write("\n")
                        f.close()
    '''
    dataset_used = "german"
    # if i == 0 and j == 2:
    protected_attribute_used = "sex"
    dataset_orig, privileged_groups, unprivileged_groups, optim_options = LoadData(dataset_used, protected_attribute_used)                
    
    l=2
    m=3
    n=0
    feature = comb_algorithm(l,m,n, dataset_orig, privileged_groups, unprivileged_groups, optim_options)
    print(l,m,n)
    print(feature)
    
    
def comb_algorithm(l, m, n, dataset_original1, privileged_groups1,unprivileged_groups1, optim_options1):
    
    dataset_original2 = copy.deepcopy(dataset_original1)
    privileged_groups2 = copy.deepcopy(privileged_groups1)
    unprivileged_groups2 = copy.deepcopy(unprivileged_groups1)
    optim_options2 = copy.deepcopy(optim_options1)
    
    print(l,m,n)
    dataset_orig_train, dataset_orig_vt = dataset_original2.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
         
    if l == 0:
        dataset_transf_train, dataset_transf_valid, dataset_transf_test = dataset_orig_train, dataset_orig_valid, dataset_orig_test
    else:
        pre_used = preAlgorithm[l-1]
        dataset_transf_train, dataset_transf_valid, dataset_transf_test = Pre(pre_used, dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups2,unprivileged_groups2, optim_options2)     
    
    
    if m == 0: 
        dataset_transf_valid_pred, dataset_transf_test_pred = train(dataset_transf_train, 
                                                                    dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2)    
    else:        
        in_used = inAlgorithm[m-1]
        if in_used == "adversarial_debiasing":
            dataset_transf_valid_pred, dataset_transf_test_pred = adversarial_debiasing(dataset_transf_train, 
                                                                                        dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2)
        elif in_used == "art_classifier":
            dataset_transf_valid_pred, dataset_transf_test_pred = art_classifier(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2)                  
        elif in_used == "prejudice_remover":
            for key,value in privileged_groups2[0].items():
                sens_attr = key
            dataset_transf_valid_pred, dataset_transf_test_pred = prejudice_remover(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2, sens_attr)


    if n == 0:
        dataset_transf_test_pred_transf = dataset_transf_test_pred
        
    else:
        post_used = postAlgorithm[n-1]
        if post_used == "calibrated_eqodds":
            cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups2,
                                     unprivileged_groups = unprivileged_groups2,
                                     cost_constraint=cost_constraint,
                                     seed=1)
            cpp = cpp.fit(dataset_transf_valid, dataset_transf_valid_pred)
            dataset_transf_test_pred_transf = cpp.predict(dataset_transf_test_pred)
                  
        elif post_used == "eqodds":
            EO = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups2, 
                                      privileged_groups=privileged_groups2, seed = 1)
            EO = EO.fit(dataset_transf_valid, dataset_transf_valid_pred)
            dataset_transf_test_pred_transf = EO.predict(dataset_transf_test_pred)
                    
        elif post_used == "reject_option":
            ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups2, 
                                 privileged_groups=privileged_groups2, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name=allowed_metrics[0],
                                  metric_ub=metric_ub, metric_lb=metric_lb)
            ROC = ROC.fit(dataset_transf_valid, dataset_transf_valid_pred)
            dataset_transf_test_pred_transf = ROC.predict(dataset_transf_test_pred)
            
            
    metric = ClassificationMetric(
                dataset_transf_test, dataset_transf_test_pred_transf,
                unprivileged_groups=unprivileged_groups2,
                privileged_groups=privileged_groups2)
    
    metrics = OrderedDict()
    metrics["Classification accuracy"] = metric.accuracy()
    TPR = metric.true_positive_rate()
    TNR = metric.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
    metrics["Balanced classification accuracy"] = bal_acc_nodebiasing_test
    metrics["Statistical parity difference"] = metric.statistical_parity_difference()
    metrics["Disparate impact"] = metric.disparate_impact()
    metrics["Equal opportunity difference"] = metric.equal_opportunity_difference()
    metrics["Average odds difference"] = metric.average_odds_difference()
    metrics["Theil index"] = metric.theil_index()
    metrics["United Fairness"] = metric.generalized_entropy_index()
    
    feature = "["
    for m in metrics:
        feature = feature + " " + str(round(metrics[m], 4)) 
    feature = feature + "]"
    
    return feature


main()        