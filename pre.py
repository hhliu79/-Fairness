#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:13:46 2019

@author: huihuiliu
"""

from aif360.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing.optim_preproc  import OptimPreproc
from aif360.algorithms.preprocessing.reweighing import Reweighing

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def Pre(algorithm_used, dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups, optim_options):

    if algorithm_used == "disparate_impact_remover":
        '''
        scaler = MinMaxScaler(copy=False)
        dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_valid.features = scaler.fit_transform(dataset_orig_valid.features)
        dataset_orig_test.features = scaler.fit_transform(dataset_orig_test.features)
        '''
        DIC = DisparateImpactRemover(repair_level=1.0)
        dataset_transf_train = DIC.fit_transform(dataset_orig_train)
        dataset_transf_valid = DIC.fit_transform(dataset_orig_valid)        
        dataset_transf_test = DIC.fit_transform(dataset_orig_test)
        
    elif algorithm_used == "lfr":
        TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups, k=5,
                 Ax=0.01,
                 Ay=1.0,
                 Az=50.0,
                 print_interval=250,
                 verbose=1,
                 seed=1)
        TR.fit(dataset_orig_train)
        dataset_transf_train = TR.transform(dataset_orig_train)        
        dataset_transf_valid = TR.transform(dataset_orig_valid)                
        dataset_transf_test = TR.transform(dataset_orig_test)
        
    
    elif algorithm_used == "optim":
        OP = OptimPreproc(OptTools, optim_options,unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
        OP.fit(dataset_orig_train)
        dataset_transf_train = OP.transform(dataset_orig_train)
        dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
        dataset_transf_valid = OP.transform(dataset_orig_valid)
        dataset_transf_valid = dataset_orig_train.align_datasets(dataset_transf_valid)
        dataset_transf_test = OP.transform(dataset_orig_test)
        dataset_transf_test = dataset_orig_train.align_datasets(dataset_transf_test)
    
    elif algorithm_used == "reweighing":
        RW = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        RW.fit(dataset_orig_train)
        dataset_transf_train = RW.transform(dataset_orig_train)
        dataset_transf_valid = RW.transform(dataset_orig_valid)
        dataset_transf_test = RW.transform(dataset_orig_test)
        
    dataset_transf_train.labels = dataset_orig_train.labels
    dataset_transf_valid.labels = dataset_orig_valid.labels
    dataset_transf_test.labels = dataset_orig_test.labels
    #dataset_transf_test.scores = dataset_orig_test.scores
    
    return dataset_transf_train, dataset_transf_valid, dataset_transf_test
