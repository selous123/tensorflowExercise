#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:09:47 2017

@author: lrh
"""

import tensorflow as tf

#%%
with tf.variable_scope("Variable"):
    a_1 = tf.Variable(0.1,"a")
    a_2 = tf.Variable(0.1,"a")

print a_1.name,a_2.name
#%%
with tf.variable_scope("get_variable"):
    b_1 = tf.get_variable(name="b",initializer=0.1)
    b_2 = tf.get_variable(name="b",initializer=0.1)

print b_1.name,b_2.name