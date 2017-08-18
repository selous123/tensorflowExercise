撰写时间：2017.8.15  

# tensorflow代码组织
tensorflow的源代码中有很多example，从那些大牛们的源代码中我们其实可以学到很多有关tensorflow的代码格式应该如何排版。但是由于tf项目组的人有很多，代码格式也是参差不齐，所以这篇博文也就是总结一些博主在实际使用tensorflow中常用的代码组织的结构。

<hr />
<font color='red'>该篇博文按照博主的代码格式变化的时间顺序组织</font>

## 简单的代码组织
 <li>tensorflow中将所有的操作都看做是节点，所有的节点组成一个前驱图，也就是tensorflow中提到的Graph计算图结构。</li>
 <li>tensorflow的另一个模块是session模块，通过session为之前建立好的Graph“喂(feed)数据”,然后从图的末端fetch需要的结果。</li>
 
 **所以最简单的代码组织就是将这两个模块分开。先建立计算图，然后通过会话模块运行计算图。**
```
#coding:utf-8
#######softmax

import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

######prepare data
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True);


#######create the graph,定义图中所有的计算节点的依赖关系
x = tf.placeholder(tf.float32,shape = [None,784],name = 'x');
y_ = tf.placeholder(tf.float32,shape = [None,10],name = 'y_');
W = tf.Variable(tf.zeros([784,10]));
b = tf.Variable(tf.zeros([10]),name = 'input_bias');
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

#set sess as default，定义会话
sess = tf.InteractiveSession();
init = tf.global_variables_initializer();
sess.run(init);
##通过会话运行图
for i in range(1000):
    batch = mnist.train.next_batch(100);
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})
    if i % 50 == 0:
    	print "Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print(accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
```	

## 使用函数
在tensorflow的官方教程：[TensorFlow Mechanics 101](https://www.tensorflow.org/get_started/mnist/mechanics)中提到
 >the graph is built from the mnist.py file according to a 3-stage pattern: inference(), loss(), and training().
 <li>inference() - Builds the graph as far as is required for running the network forward to make predictions.</li>
<li>loss() - Adds to the inference graph the ops required to generate loss.</li>
<li>training() - Adds to the loss graph the ops required to compute and apply gradients.</li>

说明在建图的过程中，如果图的结构特别复杂，所有的代码都写在一起就会非常的混乱,所以我们就可以将代码通过函数的形式分开。下面是通过函数的形式实现两层DNN+softmax的mnist手写体的多分类问题。代码如下：
```
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
```

关于如何调用，可以参考主函数的实现。[github地址](https://github.com/selous123/tensorflowExercise/blob/master/tensorflow_ex/ex1/dnn.py),代码中的run_training函数就是上面提到的training函数。

## 使用类CLASS

上述的代码组织结构的问题在于

<li>1.很难支持模型复用，也就是面向过程与面向对象的差别</li>
<li>2.没有涉及到数据的处理。因为所有的数据都是使用常用数据集，使用别人写好的api</li>

<font color="red">**所以下面的内容涉及的就是关于使用类重构代码以及写自己的数据集api**</font>

下面的内容主要是数据类和模型类中常用的一些函数，以及具体的实例代码github(代码是一个使用cnn网络识别图像的多分类问题)

### 数据类(图像数据示例)

```python
class DataSet(object):
	def __init__():
	@property
	def images():
	@property
	def labels():
	@property
	def num_examples():
	@property
	def epoches_completed():
	def next_batch():
```

[实例代码](https://github.com/selous123/malware/blob/master/dataset/load_malware.py)
### 计算model类
```
class Model(object):
	def __init__():
	@lazy_property
	def inference():
	@lazy_property
	def optimize():
	@lazy_property
	def error():
```
[实例代码](https://github.com/selous123/malware/blob/master/dataset/malware_model.py)
[主函数代码](https://github.com/selous123/malware/blob/master/dataset/main.py)
这个项目还在测试中，所以主函数的测试数据还没有喂到模型中。主要看代码结构吧


<font size=6>问题1：</font>为什么在model中使用lazy_property?
首先要明白model中的所有函数都是用来建立计算图的，所以每一个函数运行一次和运行多次，那么建立的图的结构就会不同。为lazy_property的作用就是在函数运行一次之后，第二次调用的时候就不在运行，而是直接返回结果。

```
def lazy_property(func):
    attr_name = "_lazy_" + func.__name__
    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazy_property
```
[参考地址](http://danijar.com/structuring-your-tensorflow-models/)

<font size=6>问题2：</font>主函数中的代码组织结构？
<font size=6>回答：</font>主函数中主要使用第三方包argparse解析参数，代替了tensorflow中的官方配置文件的方式。如果你是新手，可以直接使用和我代码中一样的格式，我的代码格式也是模仿的别人。
