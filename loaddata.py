#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:56:52 2019

@author: huihuiliu
"""
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas

def LoadData(dataset_used,protected_attribute_used):
    
    if dataset_used == "adult":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_original = load_preproc_data_adult(['sex'])
            
            optim_options = {
                "distortion_fun": get_distortion_adult,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        elif protected_attribute_used == "race":
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_original = load_preproc_data_adult(['race'])
            
            optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        
    
    elif dataset_used == "german":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_original = load_preproc_data_german(['sex'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
            
        elif protected_attribute_used == "age":
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            dataset_original = load_preproc_data_german(['age'])
            
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        #dataset_original = dataset_orig.copy(deepcopy=True)  
        #dataset_original.labels = dataset_original.labels - (dataset_original.labels-1.0)*2.0
        # change from 1,2 to 1,0 
        #dataset_original.labels = dataset_original.labels - (dataset_original.labels-1.0)*2.0
        dataset_original.labels = 2 - dataset_original.labels
        dataset_original.unfavorable_label = 0.0
        
    elif dataset_used == "compas":
#     dataset_orig = CompasDataset()
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 0}]
            unprivileged_groups = [{'sex': 1}]
            dataset_original = load_preproc_data_compas(['sex'])
            
            optim_options = {
                "distortion_fun": get_distortion_compas,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        
        elif protected_attribute_used == "race":
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_original = load_preproc_data_compas(['race'])
            
            optim_options = {
                "distortion_fun": get_distortion_compas,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        
    '''   
    elif dataset_used == "bank":
#     dataset_orig = CompasDataset()
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 0}]
            unprivileged_groups = [{'sex': 1}]
            dataset_original = load_preproc_data_bank(['sex'])
            
            optim_options = {
                "distortion_fun": get_distortion_compas,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        
        elif protected_attribute_used == "race":
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_original = load_preproc_data_bank(['race'])
            
            optim_options = {
                "distortion_fun": get_distortion_compas,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        
    '''   
    return dataset_original, privileged_groups, unprivileged_groups, optim_options
