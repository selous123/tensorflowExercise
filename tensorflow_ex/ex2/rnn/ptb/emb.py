#coding:utf-8
import tensorflow as tf
import numpy as np

input_data = np.arange(10).reshape(2,5)

print input_data
with tf.device("/cpu:0"):
    embedding = tf.get_variable(
                "embedding", [10, 10], dtype=np.int32)
    inputs = tf.nn.embedding_lookup(embedding,input_data )
session = tf.InteractiveSession();

session.run(tf.global_variables_initializer())
print inputs.eval().shape
#(2,5,10)
