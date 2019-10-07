#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:37:03 2019

@author: huihuiliu
"""

# Load all necessary packages
import sys
sys.path.append("../")
from aif360.metrics import BinaryLabelDatasetMetric
from loaddata import LoadData
from Original_Model import train
from pre import Pre
import numpy as np
np.random.seed(1)
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
import os
os.environ['CUDA_VISIBLE_DEVICES']='8'

dataset = ["german", "adult", "compas"]  
attribute = ["sex", "race", "age"]
preAlgorithm = ["disparate_impact_remover", "lft", "optim", "reweighing"] 
inAlgorithm = ["adversarial_debiasing", "art_classifier", "prejudice_remover"]
postAlgorithm = ["calibrated_eqodds", "eqodds", "reject_option"]    
    
# Verify metric name
allowed_metrics = ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]    
randseed = 12345679 
cost_constraint = "weighted" # "fnr", "fpr", "weighted"
metric_ub = 0.05
metric_lb = -0.05


def main():

    #f = open('/home/hliu79/firstProject/result/comb.txt', 'w')    
    f = open('/Users/huihuiliu/fairness Learning Project/result/comb_algorithm00.txt', 'a')        
    
    try:
        for i in range(3):
            dataset_used = dataset[i]
            for j in range(3):
                if i == 0 and j == 2:
                    continue
                if i == 1 and j == 1:
                    continue
                if i == 2 and j == 2:
                    continue
                if j == 0:
                    arr = dataset_used + " " + "[1, 0, 0]"
                if j == 1:
                    arr = dataset_used + " " + "[0, 1, 0]"
                if j == 2:
                    arr = dataset_used + " " + "[0, 0 ,1]"

                protected_attribute_used = attribute[j]
                dataset_original, privileged_groups, unprivileged_groups, optim_options = LoadData(dataset_used, protected_attribute_used)                
                for l in range(5):
                    for m in range(4):
                        for n in range(4):
                            algorithm_used = arr + " " + "[" + " " + str(l) + " " + str(m) + " " + str(n) + "]"
                            feature = comb_algorithm(l, m, n, dataset_original, privileged_groups, unprivileged_groups, optim_options)
                            result = str(algorithm_used + feature)
                            print(result)
                            f.write(result)
                            f.write("\n")
        f.close()
    except:   
        f.close()
        
        

def comb_algorithm(l, m, n, dataset_original, privileged_groups,unprivileged_groups, optim_options):
    
    print(l,m,n)
    dataset_orig_train, dataset_orig_vt = dataset_original.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
         
    if l == 0:
        dataset_transf_train, dataset_transf_valid, dataset_transf_test = dataset_orig_train, dataset_orig_valid, dataset_orig_test
    else:
        pre_used = preAlgorithm[l-1]
        dataset_transf = Pre(pre_used, dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups,unprivileged_groups, optim_options)
        dataset_transf_train, dataset_transf_valid, dataset_transf_test = dataset_transf
     
        
        
    if m ==0: 
        dataset_orig_valid_pred, dataset_orig_test_pred = train(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups, unprivileged_groups)    
    else:        
        in_used = inAlgorithm[m-1]
        for key,value in privileged_groups[0].items():
                    sens_attr = key
        if in_used == "adversarial_debiasing":
            dataset_orig_valid_pred, dataset_orig_test_pred = adversarial_debiasing(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups, unprivileged_groups, True)
        elif in_used == "art_classifier":
            dataset_orig_valid_pred, dataset_orig_test_pred = art_classifier(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups, unprivileged_groups)                  
        elif in_used == "prejudice_remover":
            dataset_orig_valid_pred, dataset_orig_test_pred = prejudice_remover(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups, unprivileged_groups, sens_attr, 25.0)



    if n == 0:
        dataset_transf_test_pred = dataset_orig_test_pred
        
    else:
        post_used = postAlgorithm[n-1]
        if post_used == "calibrated_eqodds":
            cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=randseed)
            cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
            dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
                  
        elif post_used == "eqodds":
            EO = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, 
                                      privileged_groups=privileged_groups, seed = randseed)
            EO = EO.fit(dataset_orig_valid, dataset_orig_valid_pred)
            dataset_transf_test_pred = EO.predict(dataset_orig_test_pred)
                    
        elif post_used == "reject_option":
            ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name=allowed_metrics[0],
                                  metric_ub=metric_ub, metric_lb=metric_lb)
            ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
            dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

    metric = ClassificationMetric(
                dataset_orig_test, dataset_transf_test_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    
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
    print(metrics)

    
    feature = "["
    for m in metrics:
        feature = feature + " " + str(round(metrics[m], 4)) 
    feature = feature + "]"
    
    return feature
        
        
main()        