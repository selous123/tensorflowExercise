#coding:utf-8

#######softmax
import tensorflow as tf
import numpy as np 



feed_a = tf.constant([2,3]);

a = tf.placeholder(tf.float32,shape = [None]);

op = tf.square(a);

sess = tf.InteractiveSession();

#tranform tensor into numpy
feed = feed_a.eval(session = sess);

print type(feed);
print op.eval(session = sess,feed_dict={a:feed});


