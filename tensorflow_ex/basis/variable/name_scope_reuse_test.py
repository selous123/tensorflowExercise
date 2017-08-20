#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:45:05 2017

@author: lrh
"""

import tensorflow as tf

def inference():
    with tf.variable_scope("conv"):
        weight = tf.get_variable("weight",initializer=1.0)

    return weight

with tf.variable_scope("compute") as scope:   
    output = inference()
    scope.reuse_variables()
    output2 = inference()

print output.name,output2.name

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(output)