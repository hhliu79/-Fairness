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

def Pre(algorithm_used, dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups, optim_options):

    if algorithm_used == "disparate_impact_remover":
        DIC = DisparateImpactRemover()
        dataset_transf_train = DIC.fit_transform(dataset_orig_train)
        dataset_transf_valid = DIC.fit_transform(dataset_orig_valid)        
        dataset_transf_test = DIC.fit_transform(dataset_orig_test)
        
    elif algorithm_used == "lfr":
        TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
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
    
    dataset_transf = [dataset_orig_train, dataset_orig_valid, dataset_orig_test]
    return dataset_transf


