#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:54:30 2019

@author: huihuiliu
"""

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='8'

def adversarial_debiasing(dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups, debias):
    
    tf.reset_default_graph()
    sess = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='plain_classifier',
                          debias=False,
                          sess=sess)
    plain_model.fit(dataset_orig_train)
    
    dataset_nodebiasing_valid = plain_model.predict(dataset_orig_valid)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)
    print(dataset_nodebiasing_test)
    sess.close()
       
    return dataset_nodebiasing_valid, dataset_nodebiasing_test

