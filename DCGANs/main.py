#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import cv2
from dcganModel import DCGANsModel
FLAGS = None

def sample(size):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=size)

def training():
    #prepare mnist data
    data_set = input_data.read_data_sets(FLAGS.input_data_dir);

    z = tf.placeholder(tf.float32,[FLAGS.batch_size,100],name = "Z")
    x = tf.placeholder(tf.float32,[FLAGS.batch_size,784],name = "input")
    
    dcgansM = DCGANsModel(x,z)
    
    G_sample,_,_ = dcgansM.inference
    D_loss,G_loss,D_optimizer,G_optimizer = dcgansM.optimize
    #= dcgansM.loss
    
    

    
    merged_summary_op = tf.summary.merge_all()
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir);
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir,session.graph)
    z_sample = sample([FLAGS.batch_size,100]);
    for step in range(FLAGS.max_steps):
        x_input,_ = data_set.train.next_batch(FLAGS.batch_size)
        
        for d_step in range(FLAGS.d_steps):
            _,D_loss_curr= session.run([D_optimizer,D_loss],feed_dict ={
                x:x_input,
                z:z_sample
                })
        for g_step in range(FLAGS.g_steps):
            _,G_loss_curr = session.run([G_optimizer,G_loss],feed_dict ={
                z:z_sample
                })
        if step%100==0:
            print "step:{},D_loss:{},G_loss:{}".format(step,D_loss_curr,G_loss_curr)
            summary_str = session.run(merged_summary_op,feed_dict={x: x_input, z:z_sample});
            summary_writer.add_summary(summary_str, step);
            #print D_real_curr
        if step%100 == 0:
            G_sample_curr = session.run([G_sample],feed_dict={
                    z:z_sample
                    })
            img = np.array(G_sample_curr).reshape(-1,28,28)
            img_dir = str(step/100)
            if tf.gfile.Exists(FLAGS.output_dir+img_dir):
                tf.gfile.DeleteRecursively(FLAGS.output_dir+img_dir);
            tf.gfile.MakeDirs(FLAGS.output_dir+img_dir);
            for ind in range(FLAGS.batch_size):
                cv2.imwrite(FLAGS.output_dir+img_dir+"/"+str(ind)+".jpg",img[ind]*255)
def main(_):
    training()


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_data_dir',
        type = str,
        default = "/home/lrh/dataset/mnist",
        help = "directory of mnist data"
        )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help="batch size"
        )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=100000,
        help='max steps'
        )
    
    parser.add_argument(
        '--g_steps',
        type=int,
        default=5,
        help='Generator steps'
        )
    parser.add_argument(
        '--d_steps',
        type=int,
        default=1,
        help='Discriminator steps'
        )
    parser.add_argument(
        '--log_dir',
        type=str,
        default="/tmp/gan/log",
        help='the log directory'
        )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="/tmp/gan/mnist/",
        help="output directory")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)