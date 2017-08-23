#coding:utf-8


import tensorflow as tf

a = tf.Variable(tf.truncated_normal(shape=[5]),collections=["train"],name="g_a")
###build graph
init = tf.global_variables_initializer()
print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="^g_")
print tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
print tf.get_collection("train")
print tf.get_collection_ref("train")
sess = tf.Session()
sess.run(init)



