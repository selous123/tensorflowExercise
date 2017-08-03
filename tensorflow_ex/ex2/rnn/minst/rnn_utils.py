#!/usr/bin/env python2
#coding: utf-8
"""
Created on Sun May 28 11:41:40 2017

@author: lrh
"""

import tensorflow as tf
from tensorflow.contrib import rnn

input_size = 28;#num_unit of lstm cell;
n_steps = 28;#timestamp
n_hidden = 128;#hidden layer size of lstm cell
n_classes = 10;

FLAGS = None;
#forward propagation.
def inference(images , model='lstm'):
    
    """
    Args:
        images:[batch_size,IMAGE_SIZE,IMAGE_SIZE];
    """
    if model == 'rnn':
        cell_fun = rnn.BasicRNNCell         
    elif model == 'gru':
        cell_fun = rnn.GRUCell
    elif model == 'lstm':
        cell_fun = rnn.BasicLSTMCell
    
    with tf.name_scope('input'):
        images = tf.reshape(images,[-1,n_steps,input_size],name = 'reshape');
    #rnn
    with tf.name_scope("rnn"):
        cell = cell_fun(n_hidden, state_is_tuple=True)
        image_split = tf.unstack(images,n_steps,1);
        outputs, states = rnn.static_rnn(cell, image_split, dtype=tf.float32)  
        #时间序列上每个Cell的输出:[... shape=(128, 28)..]
    
    #lr
    with tf.name_scope("lr"):
        
        weights = tf.Variable(tf.random_normal([n_hidden, n_classes]),name='weights')
        bias = tf.Variable(tf.zeros([n_classes]),name = 'bias');
        # Linear activation
        # Get the last output
        return tf.matmul(outputs[-1], weights) + bias

        #, lstm.state_size # State size to initialize the stat
    
    """
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
    
    initial_state = cell.zero_state(batch_size, tf.float32)
 
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
        softmax_b = tf.get_variable("softmax_b", [len(words)+1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)
 
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs,[-1, rnn_size])
 
    logits = tf.matmul(output, softmax_w) + softmax_b
     probs = tf.nn.softmax(logits)
     return logits, last_state, probs, cell, initial_state
 
    """
    

#loss function
def loss(logits,labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

#train
def training(loss,learning_rate):
    
    tf.summary.scalar('loss',loss);
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss);
    return train_op;

def evaluation(logits,label):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy;
