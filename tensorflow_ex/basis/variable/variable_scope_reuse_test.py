# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

#==============================================================================
# with tf.variable_scope("var") as scope:
#     v = tf.get_variable("t",initializer=1.0)
#     v2 = tf.get_variable("t",initializer=1.0,reuse=True)
#==============================================================================

#reuse attribution belongs to variable_scope() method
#not get_variables() method
#==============================================================================
# with tf.variable_scope("foo") as scope:
#     v = tf.get_variable("v",initializer=1.0)
#     scope.reuse_variables()
#     v1 = tf.get_variable("v",initializer=1.0)
# 
#==============================================================================
#reuse attribution is set
with tf.variable_scope("foo",reuse=True):
    v2 = tf.get_variable("v",initializer=1.0)
    
    
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


print sess.run(v2)
