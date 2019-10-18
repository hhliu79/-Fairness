#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:44:55 2019

@author: huihuiliu
"""

import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
#from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from inprocess1 import CalibratedEqOddsPostprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
cost_constraint = "fnr" # "fnr", "fpr", "weighted"

def calibrated_eqodd(dataset_orig_valid, dataset_orig_valid_pred, dataset_orig_test,dataset_orig_test_pred, privileged_groups, unprivileged_groups):

    # Learn parameters to equalize odds and apply to create a new dataset
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint)
    cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
    
    
    #dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
    dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

    return dataset_transf_test_pred
      
    

