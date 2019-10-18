#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:38:14 2019

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
from Ad_Debiasing import adversarial_debiasing
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
from PlainModel import plain_model
import os
os.environ['CUDA_VISIBLE_DEVICES']='9'

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
cost_constraint = "fpr" # "fnr", "fpr", "weighted"
metric_ub = 0.05
metric_lb = -0.05
       

def main():
    
    dataset_used = "german"
    protected_attribute_used = "sex"
    arr = dataset_used + " " + "[1, 0, 0]"
    data = np.zeros((80,8))
    i = 0
    
    for l in range(5):
        for m in range(4):
            for n in range(4):
                dataset_orig, privileged_groups, unprivileged_groups, optim_options = LoadData(dataset_used, protected_attribute_used)                
                algorithm_used = arr + " " + "[" + " " + str(l) + " " + str(m) + " " + str(n) + "]"
                while(True):
                    feature, feature_str = comb_algorithm(l, m, n, dataset_orig, privileged_groups, unprivileged_groups, optim_options)                    
                    if feature[1] == 0.5 :
                        print(feature)
                        continue    
                    else:
                        break
                        print(feature)
                data[i] = feature
                i = i+1               
                result = str(algorithm_used + feature_str)
                print(result)
                with open('german_sex_forAcc.txt', 'a') as f:
                    f.write(result)
                    f.write("\n")
                    f.close()
    '''
    dataset_used = "german"
    # if i == 0 and j == 2:
    protected_attribute_used = "sex"
    dataset_orig, privileged_groups, unprivileged_groups, optim_options = LoadData(dataset_used, protected_attribute_used)                
    
    l=0
    m=1
    n=1
    feature, feature_str = comb_algorithm(l,m,n, dataset_orig, privileged_groups, unprivileged_groups, optim_options)
    print(l,m,n)
    print(feature)
    '''        
    
def comb_algorithm(l, m, n, dataset_original1, privileged_groups1,unprivileged_groups1, optim_options1):

    dataset_original2 = copy.deepcopy(dataset_original1)
    privileged_groups2 = copy.deepcopy(privileged_groups1)
    unprivileged_groups2 = copy.deepcopy(unprivileged_groups1)
    optim_options2 = copy.deepcopy(optim_options1)
    
    print(l,m,n)
    dataset_original_train, dataset_original_vt = dataset_original2.split([0.7], shuffle=True)
    dataset_original_valid, dataset_original_test = dataset_original_vt.split([0.5], shuffle=True)
    dataset_original_test.labels = dataset_original_test.labels
    print('=======================')
    #print(dataset_original_test.labels)
    dataset_orig_train = copy.deepcopy(dataset_original_train)
    dataset_orig_valid = copy.deepcopy(dataset_original_valid)
    dataset_orig_test = copy.deepcopy(dataset_original_test)
    
   
    if l == 0:
        dataset_transfer_train = copy.deepcopy(dataset_original_train)
        dataset_transfer_valid = copy.deepcopy(dataset_original_valid)
        dataset_transfer_test = copy.deepcopy(dataset_original_test)
        #dataset_transf_train, dataset_transf_valid, dataset_transf_test = dataset_orig_train, dataset_orig_valid, dataset_orig_test
    else:
        pre_used = preAlgorithm[l-1]
        dataset_transfer_train, dataset_transfer_valid, dataset_transfer_test = Pre(pre_used, dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups2,unprivileged_groups2, optim_options2)     
    

    
    dataset_transf_train = copy.deepcopy(dataset_transfer_train)
    dataset_transf_valid = copy.deepcopy(dataset_transfer_valid)
    dataset_transf_test = copy.deepcopy(dataset_transfer_test)
    if m == 0: 
        dataset_transfer_valid_pred, dataset_transfer_test_pred = plain_model(dataset_transf_train, 
                                                                    dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2)    
    else:        
        in_used = inAlgorithm[m-1]
        if in_used == "adversarial_debiasing":
            dataset_transfer_valid_pred, dataset_transfer_test_pred = adversarial_debiasing(dataset_transf_train, 
                                                                                        dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2)
        elif in_used == "art_classifier":
            dataset_transfer_valid_pred, dataset_transfer_test_pred = art_classifier(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2)                  
        elif in_used == "prejudice_remover":
            for key,value in privileged_groups2[0].items():
                sens_attr = key
            dataset_transfer_valid_pred, dataset_transfer_test_pred = prejudice_remover(dataset_transf_train, dataset_transf_valid, dataset_transf_test, privileged_groups2, unprivileged_groups2, sens_attr)
    


    dataset_transf_valid_pred = copy.deepcopy(dataset_transfer_valid_pred)
    dataset_transf_test_pred = copy.deepcopy(dataset_transfer_test_pred)
    if n == 0:
        dataset_transf_test_pred_transf = copy.deepcopy(dataset_transfer_test_pred)
        
    else:
        post_used = postAlgorithm[n-1]
        if post_used == "calibrated_eqodds":
            cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups2,
                                     unprivileged_groups = unprivileged_groups2,
                                     cost_constraint=cost_constraint)
            cpp = cpp.fit(dataset_transfer_valid, dataset_transf_valid_pred)
            dataset_transf_test_pred_transf = cpp.predict(dataset_transf_test_pred)
                  
        elif post_used == "eqodds":
            EO = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups2, 
                                      privileged_groups=privileged_groups2)
            EO = EO.fit(dataset_transfer_valid, dataset_transf_valid_pred)
            dataset_transf_test_pred_transf = EO.predict(dataset_transf_test_pred)
                    
        elif post_used == "reject_option":
            #dataset_transf_test_pred_transf = reject_option(dataset_transf_valid, dataset_transf_valid_pred, dataset_transf_test, dataset_transf_test_pred, privileged_groups2, unprivileged_groups2)
            
            ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups2, 
                                 privileged_groups=privileged_groups2)
            ROC = ROC.fit(dataset_transfer_valid, dataset_transf_valid_pred)
            dataset_transf_test_pred_transf = ROC.predict(dataset_transf_test_pred)
            
    #print('=======================')
    org_labels = dataset_orig_test.labels
    print(dataset_orig_test.labels)
    #print(dataset_transf_test.labels)
    #print('=======================')
    pred_labels = dataset_transf_test_pred.labels
    print(dataset_transf_test_pred.labels)
    
    true_pred = org_labels == pred_labels
    print("acc after in: ", float(np.sum(true_pred))/pred_labels.shape[1])
    #print('=======================')
    #print(dataset_transf_test_pred_transf.labels)
    #print(dataset_transf_test_pred_transf.labels.shape)

    metric = ClassificationMetric(
                dataset_transfer_test, dataset_transf_test_pred_transf,
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
    
    feature = []
    feature_str = "["
    for m in metrics:
        data = round(metrics[m], 4)
        feature.append(data)
        feature_str = feature_str + str(data) + " "
    feature_str = feature_str + "]"

    return feature, feature_str

    
main()        