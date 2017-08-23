#coding:utf-8

import tensorflow as tf


#==============================================================================
# def batch_norm_layer(x, train_phase, scope_bn):
#     with tf.variable_scope(scope_bn):
#         beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
#         axises = np.arange(len(x.shape) - 1)
#         batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)
# 
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
# 
#         mean, var = tf.cond(train_phase, mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed
# 
#==============================================================================

def batch_norm_layer(input,name=None):
    """
    Args:
        input:4-d tensor
        
    """
    mean,var = tf.nn.moments(input,[0,1,2])
    gamma = tf.get_variable(name="gamma",shape=[1],initializer=tf.constant_initializer(0.0)) 
    beta = tf.get_variable(name="beta",shape=[1],initializer=tf.constant_initializer(0.0))
    variance_const = tf.get_variable(name="const",shape=[1],initializer=tf.constant_initializer(0.1),trainable=False)
    return tf.nn.batch_normalization(input,mean,var,beta,gamma,variance_const,name="batch_normal")
    


def inference(images):
    """
    compute forward compute,and build compute graph.
    Args:
        images_placeholder:size(images_num,784);
    Return: 
        logits:images_num*NUMCLASS,predict result of 
        the network.
    """
    with tf.name_scope('input'):
        images_input = tf.reshape(images,[-1,28,28,1],name = 'reshape');

    with tf.name_scope('conv1'):
        #filter
        weight = tf.Variable(tf.truncated_normal(shape = [5,5,1,32],stddev=0.1),name = 'weight');
        bias = tf.Variable(tf.constant(0.1,shape = [32]));
        conv1 = tf.nn.relu(batch_norm_layer(tf.nn.conv2d(images_input,weight,strides=[1, 1, 1, 1], padding='SAME')+bias))
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


    with tf.name_scope('conv2'):
        weight = tf.Variable(tf.truncated_normal(shape = [5,5,32,64],stddev=0.1),name = 'weight');
        bias = tf.Variable(tf.constant(0.1,shape = [64]));
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1,weight,strides=[1, 1, 1, 1], padding='SAME')+bias)
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')



    with tf.name_scope('nn1'):
        pool2_flat = tf.reshape(pool2,[-1,7*7*64]);
        weight = tf.Variable(tf.truncated_normal(shape = [7*7*64,1024],stddev = 0.01),name = 'weight');
        bias = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'bias');
        nn1_output = tf.nn.relu(tf.matmul(pool2_flat,weight)+bias);


    with tf.name_scope('softmax_layer'):
        weight = tf.Variable(tf.truncated_normal(shape =[1024,10],stddev = 0.01),name ='weight');
        bias = tf.Variable(tf.constant(0.1,shape =[10]),name = 'bias');
        logits = tf.nn.softmax(tf.matmul(nn1_output,weight)+bias);

    return logits;



def loss(logits,labels):
    """
    Args:
        logits:has been normalized with softmax of all samples.
            size:[SAMPLE,numclass]
        labels:true label of all samples.size:[SAMPLES,numclass].
    Return:
        Loss tensor of type float
    """
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits));
    return cross_entropy;


def training(loss,learning_rate):

    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """

    tf.summary.scalar('loss',loss);
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op;


def evaluation(logits,labels):

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy;



