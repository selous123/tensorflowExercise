#coding:utf-8

from __future__ import division

import tensorflow as tf 
import numpy as np

import argparse
import os
import sys

import read_housing as rh

FLAGS = None


def inference(data):
    with tf.name_scope('linearReg'):
        weight = tf.Variable(tf.constant(0.1,shape = [FLAGS.dimension,1]),name = 'weights'); 
        bias = tf.Variable(tf.zeros(1),name = 'bias');
        logits = tf.matmul(data,weight)+bias;
        return logits;

def loss(labels,logits):
    loss = tf.sqrt(tf.reduce_mean(tf.square(labels - logits)));
    return loss;


def training(loss,learning_rate):

    with tf.name_scope('loss'):
        tf.summary.scalar('loss_func',loss);
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss);
    return train_op;

def fill_feed_dict(data_set,data_placeholder,labels_placeholder):

    feed_dict = {
        data_placeholder:data_set['data'],
        labels_placeholder:data_set['labels']
    }
    return feed_dict;

def train_running():

    train_set,test_set=rh.read_data_housing();
    FLAGS.dimension = train_set['data'].shape[1];

    data = tf.placeholder(tf.float32,shape = [None,FLAGS.dimension]);
    labels = tf.placeholder(tf.float32,shape = [None]);

    #logits:prefict value.
    logits = inference(data);
    square_loss = loss(labels,logits);
    train_op = training(square_loss,FLAGS.learning_rate);
    init = tf.global_variables_initializer();
    summary = tf.summary.merge_all();

    sess = tf.Session();
    summary_writer = tf.summary.FileWriter(FLAGS.logdir,sess.graph);
    sess.run(init);
    for step in range(1000):
        feed_dict = fill_feed_dict(train_set,data,labels);
        sess.run(train_op,feed_dict = feed_dict);
        if(0==step%100):
            print ("square loss %d: %f" %(step,square_loss.eval(session = sess,feed_dict = feed_dict)));
            summary_str = sess.run(summary,feed_dict = feed_dict);
            summary_writer.add_summary(summary_str,step);
            summary_writer.flush();

    feed_dict = fill_feed_dict(test_set,data,labels)

#    print logits.eval(session = sess,feed_dict = feed_dict);
    print ("square loss : %f" %(square_loss.eval(session = sess,feed_dict = feed_dict)));


def main(_):
    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir);
    tf.gfile.MakeDirs(FLAGS.logdir);
    train_running();


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir',
        type = str,
        default = '/tmp/tensorflow/hoston/logs/linear_reg',
        help = 'Directory of tensorboard log file'
        )

    parser.add_argument(
        '--learning_rate',
        type = float,
        default = 3e-6,
        help = 'learning rate')

    parser.add_argument(
        '--dimension',
        type = int,
        help ='Dimension of samples')


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



