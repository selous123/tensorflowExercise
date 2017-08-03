import tensorflow as tf

import math

# The output classes.representing the digits from 0 to 9
NUMCLASS = 10;

## 
IMAGE_SIZE = 28;
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE;


def inference(images,hidden1_unit,hidden2_unit):
    """build the graph
    
    Args:
      images:Image placeholder.Size[batchsize*IMAGE_PIEXLS]
      hidden1_unit:Size of first hidden layer.
      hidden2_unit:Size of second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    #hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden1_unit],
            stddev = 1/math.sqrt(float(hidden2_unit))),name = 'weights');
        bias = tf.Variable(tf.zeros([hidden1_unit]),name = 'bias');
        hidden1 = tf.nn.relu(tf.matmul(images,weights)+bias);


    #hidden2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_unit,hidden2_unit],
            stddev = 1/math.sqrt(float(hidden2_unit))),name = 'weights');
        bias = tf.Variable(tf.zeros([hidden2_unit]),name = 'bias');
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights)+bias);

    #softmax layer
    with tf.name_scope('softmax_layer'):
        weights = tf.Variable(tf.truncated_normal([hidden2_unit,NUMCLASS],
            stddev = 1/math.sqrt(float(hidden2_unit))),name ='weights');
        bias = tf.Variable(tf.zeros([NUMCLASS]),name = 'bias');
        logits = tf.matmul(hidden2,weights)+bias;
    return logits;



def loss(labels,logits):

    """Calculates the loss from the logits and the labels.

    Args:
    labels: Labels tensor, int32 - [batch_size].
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].

    Return: Loss tensor of type float

    """
    labels = tf.to_int64(labels)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


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
    return train_op



def evaluation(labels,logits):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    #cast(x,dtype),transform x into dtype.
    return tf.reduce_sum(tf.cast(correct, tf.int32))