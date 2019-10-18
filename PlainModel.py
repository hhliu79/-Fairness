#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:51:50 2019

@author: huihuiliu
"""

from plain_model import PlainModel
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from aif360.metrics import ClassificationMetric
import os
os.environ['CUDA_VISIBLE_DEVICES']='9'
import numpy as np
from collections import defaultdict
import copy


def plain_model(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups):
    
    dataset_original_train = copy.deepcopy(dataset_orig_train)
    dataset_original_valid = copy.deepcopy(dataset_orig_valid)
    dataset_original_test = copy.deepcopy(dataset_orig_test)
    
    tf.reset_default_graph()
    sess = tf.Session()
    debised_classifier = PlainModel(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='plain_classifier',
                          sess=sess)
    debised_classifier.fit(dataset_original_train)
    
    dataset_nodebiasing_valid = debised_classifier.predict(dataset_original_valid)
    dataset_nodebiasing_test = debised_classifier.predict(dataset_original_test)
    sess.close()
       
    return dataset_nodebiasing_valid, dataset_nodebiasing_test