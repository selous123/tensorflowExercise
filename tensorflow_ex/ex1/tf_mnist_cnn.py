#coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import cnn_network as cnn_mnist
FLAGS = None;

NUMCLASS = 10;

IMAGESIZE = 28;
IMAGEPIXELS = IMAGESIZE * IMAGESIZE;

def placeholder_inputs():

    images_placeholder = tf.placeholder(tf.float32,shape = (None,IMAGEPIXELS));
    labels_placeholder = tf.placeholder(tf.float32,shape = (None,NUMCLASS));
    return images_placeholder,labels_placeholder


def fill_feed_dict(data_set,images_pl,labels_pl):

    images,labels = data_set.next_batch(FLAGS.batch_size);

    feed_dict = {
        images_pl : images,
        labels_pl : labels
    }
    return feed_dict;


def train_runing():

    #read data_set using tensorflow providing tools.
    data_set = input_data.read_data_sets(FLAGS.input_data_dir,one_hot=True);
    print type(data_set);
    raw_input("wait");
    #####build compute graph.
    images_placeholder,labels_placeholder = placeholder_inputs();
    logits = cnn_mnist.inference(images_placeholder);
    loss = cnn_mnist.loss(logits,labels_placeholder);
    train_op = cnn_mnist.training(loss,FLAGS.learning_rate);
    accuracy = cnn_mnist.evaluation(logits,labels_placeholder);
    summary = tf.summary.merge_all();
    init = tf.global_variables_initializer();

    ###build section and some utils.
    sess = tf.InteractiveSession();
    sess.run(init);
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph);

    for i in range(10000):
        feed_dict = fill_feed_dict(data_set.train,images_placeholder,labels_placeholder);
        if i%100 == 0:
            train_accuracy,p = sess.run([accuracy,logits],feed_dict=feed_dict)
            print("step:{}, training accuracy:{},prediction:{}").format(i, train_accuracy,p)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
        train_op.run(feed_dict=feed_dict)

    print("test accuracy %g"%accuracy.eval(feed_dict={images_placeholder: data_set.test.images,
        labels_placeholder: data_set.test.labels}))



def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir);
    tf.gfile.MakeDirs(FLAGS.log_dir);
    train_runing();



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