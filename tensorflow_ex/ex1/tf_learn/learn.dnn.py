#coding:utf-8

from __future__ import division

import tensorflow as tf
import numpy as np

import argparse
import sys
import os
import urllib
#Data Set
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

FLAGS = None

FEATURE_NUM = 4;
NUMCLASS = 3;
def placeholder_inputs():
    """
    Return:
        plcaholder of input and labels.
    """
    input_placeholder = tf.placeholder(tf.float32,shape = [None,FEATURE_NUM]);
    labels_placeholder = tf.placeholder(tf.int32,shape = [None]);
    return input_placeholder,labels_placeholder;

def inference(input_placeholder,hidden):

    """
    build graph and compute perdict result.

    Args:
        input_placeholder:
    Return:
        logits:

    """
    with tf.name_scope('nn1'):
        weights = tf.Variable(tf.truncated_normal(shape = [FEATURE_NUM,hidden[0]],stddev = 0.1) ,name = 'weights')
        bias = tf.Variable(tf.constant(0.1,shape = [hidden[0]]),name = 'bias');
        nn1_output = tf.nn.relu(tf.matmul(input_placeholder,weights)+bias);

    with tf.name_scope('nn2'):
        weights = tf.Variable(tf.truncated_normal(shape = [hidden[0],hidden[1]],stddev = 0.1) ,name = 'weights')
        bias = tf.Variable(tf.constant(0.1,shape = [hidden[1]]),name = 'bias');
        nn2_output = tf.nn.relu(tf.matmul(nn1_output,weights)+bias);

    with tf.name_scope('nn3'):
        weights = tf.Variable(tf.truncated_normal(shape = [hidden[1],hidden[2]],stddev = 0.1) ,name = 'weights')
        bias = tf.Variable(tf.constant(0.1,shape = [hidden[2]]),name = 'bias');
        nn3_output = tf.nn.relu(tf.matmul(nn2_output,weights)+bias);

    with tf.name_scope('softmax_layer'):
        weights = tf.Variable(tf.truncated_normal(shape =[hidden[2],NUMCLASS],stddev = 0.1) ,name = 'weights')
        bias = tf.Variable(tf.constant(0.1,shape = [NUMCLASS]),name = 'bias');
        output = tf.nn.softmax(tf.matmul(nn3_output,weights)+bias);

    return output;


def loss(logits,labels):
    """
    Args:
        logits: logits tensor,[SAMPLES,NUMCLASS]
        labels: labels tensor,[SAMPLES,NUMCLASS]
    Return: 
        Loss tensor of type float
    """

    labels = tf.to_int64(labels);
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = labels,name = 'xentropy');
        loss = tf.reduce_mean(cross_entropy,name = 'xentropy_mean');
    return loss;


def training(loss,learning_rate):
    """
    Args:

    Return:

    """
    tf.summary.scalar("loss",loss);
    optimizer = tf.train.GradientDescentOptimizer(learning_rate);
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op;

def evaluation(logits,labels):

    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    #cast(x,dtype),transform x into dtype.
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def run_training():

    # Load datasets.
    # training_set.data   training_set.target
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)


    #build compute graph.
    input_placeholder,labels_placeholder = placeholder_inputs();
    hidden = [10,20,10];
    logits = inference(input_placeholder,hidden);
    cross_loss = loss(logits,labels_placeholder);
    train_op = training(cross_loss,FLAGS.learning_rate);
    accuracy = evaluation(logits,labels_placeholder);
    init = tf.global_variables_initializer();
    summary = tf.summary.merge_all();

    ##create session
    sess = tf.InteractiveSession();
    sess.run(init);
    summary_writer = tf.summary.FileWriter(FLAGS.logdir,sess.graph);

    print training_set.data.shape;

    for step in range(FLAGS.max_steps):
        feed_dict = {
            input_placeholder:training_set.data,
            labels_placeholder:training_set.target
            }
        train_op.run(feed_dict=feed_dict);
        if step%100 == 0:
            train_accuracy = accuracy.eval(feed_dict = feed_dict);
            print ("accuracy :%f in step %d"%(train_accuracy,step));
            summary_str = sess.run(summary,feed_dict);
            summary_writer.add_summary(summary_str,step);
            summary_writer.flush();

    print "test accuracy : %f"%accuracy.eval(feed_dict = {
            input_placeholder:test_set.data,
            labels_placeholder:test_set.target
            })

def main(_):
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "w") as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "w") as f:
            f.write(raw)
    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir);
    tf.gfile.MakeDirs(FLAGS.logdir);
    run_training();




if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument(
        '--logdir',
        type = str,
        default = '/tmp/tensorflow/iris/logs/learn_nn',
        help = 'Directory of log information'
        );

    parser.add_argument(
        '--learning_rate',
        type = float,
        default =0.1,
        help = 'learning rate'
        )

    parser.add_argument(
        '--max_steps',
        type =int,
        default = 2000,
        help = 'Max Step'
        )

FLAGS,unparse = parser.parse_known_args();
tf.app.run(main=main,argv = [sys.argv[0]]+unparse);
