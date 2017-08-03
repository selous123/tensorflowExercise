#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:41:40 2017

@author: lrh
"""
 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import rnn_utils as utils
import sys

input_size = 28;#num_unit of lstm cell;
n_steps = 28;#timestamp
n_hidden = 128;#hidden layer size of lstm cell
n_classes = 10;

FLAGS = None;

def placeholder_inputs():

    images_placeholder = tf.placeholder(tf.float32,shape = (None,input_size*n_steps));
    labels_placeholder = tf.placeholder(tf.float32,shape = (None,n_classes));
    return images_placeholder,labels_placeholder


def fill_feed_dict(data_set,images_pl,labels_pl):

    images,labels = data_set.next_batch(FLAGS.batch_size);

    feed_dict = {
        images_pl : images,
        labels_pl : labels
    }
    return feed_dict;

    
#train running    
def train_running():
    #read data_set using tensorflow providing tools.
    data_set = input_data.read_data_sets(FLAGS.input_data_dir,one_hot=True);

    #####build compute graph.
    
    # tf Graph input
    images,labels = placeholder_inputs()
    logits = utils.inference(images);
    loss = utils.loss(logits,labels);
    train_op = utils.training(loss,FLAGS.learning_rate);
    accuracy = utils.evaluation(logits,labels);
    summary = tf.summary.merge_all();
    init = tf.global_variables_initializer();

    ###build section and some utils.
    sess = tf.InteractiveSession();
    sess.run(init);
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph);

    for i in range(10000):
        feed_dict = fill_feed_dict(data_set.train,images,labels);
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("step {}, training accuracy {}".format(i, train_accuracy))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
        train_op.run(feed_dict=feed_dict)

    print("test accuracy %g"%accuracy.eval(feed_dict={images: data_set.test.images,
        labels: data_set.test.labels}))

    
    
#main function
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir);
    train_running()
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser();
    parser.add_argument(
        '--input_data_dir',
        type = str,
        default='MNIST_data/',
        help = 'Directory of Input Data.'
        )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/cnn_mnist',
        help='Directory to put the log data.'
        )
    parser.add_argument(
        '--learning_rate',
        type = float,
        default='1e-4',
        help = 'learning rate of optimizer'
        )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = '50',
        help = 'batch size'
        )
    parser.add_argument(
        '--checkpoint_dir',
        type = str,
        default = '/tmp/tensorflow/mnist/checkpoint/cnn_mnist',
        help = 'Directory to put the checkpoint data.'
        )
    FLAGS,unparse = parser.parse_known_args();
    tf.app.run(main=main,argv=[sys.argv[0]]+unparse);
    