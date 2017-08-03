#coding:utf-8

#######softmax
import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data


import argparse
import sys

FLAGS = None;


def main(_):

    ######prepare data
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot = True);


    #######create the graph
    x = tf.placeholder(tf.float32,shape = [None,784],name = 'x');
    #initialize weights and bias;
    #with tf.name_scope('input_weight'):
    W = tf.Variable(tf.zeros([784,10]));
    #with tf.name_scope('input_bias'):
    b = tf.Variable(tf.zeros([10]),name = 'input_bias');
    #Predict class and loss function
    #with tf.name_scope('y'):
    y = tf.nn.softmax(tf.matmul(x,W) + b)



    #compute loss and optimizer
    y_ = tf.placeholder(tf.float32,shape = [None,10],name = 'y_');
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_));
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_))
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    
    #####optimization:梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



    #Cannot evaluate tensor using `eval()`: No default session is registered. 
    #Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`

    #set sess as default
    sess = tf.InteractiveSession();
    ########create session
    #sess = tf.Session();
    #init op 
    init = tf.global_variables_initializer();
    sess.run(init);


    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(100);
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]});

        if i % 50 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
            print "Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


    ########evaluate
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__=='__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('--data_dir', type=str, default='MNIST_data/',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)