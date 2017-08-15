撰写时间：2017.8.15  

# tensorflow代码组织
tensorflow的源代码中有很多example，从那些大牛们的源代码中我们其实可以学到很多有关tensorflow的代码格式应该如何排版。但是由于tf项目组的人有很多，代码格式也是参差不齐，所以这篇博文也就是总结一些博主在实际使用tensorflow中常用的代码组织的结构。

<hr />
<font color='red'>该篇博文按照博主的代码格式变化的时间顺序组织</font>

## 简单的代码组织
 <li>tensorflow中将所有的操作都看做是节点，所有的节点组成一个前驱图，也就是tensorflow中提到的Graph计算图结构。</li>
 <li>tensorflow的另一个模块是session模块，通过session为之前建立好的Graph“喂(feed)数据”,然后从图的末端fetch需要的结果。</li>
 
 **所以最简单的代码组织就是将这两个模块分开。先建立计算图，然后通过会话模块运行计算图。**
 
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
	
	

## 使用函数
在tensorflow的官方教程：[TensorFlow Mechanics 101][0]中提到
 >the graph is built from the mnist.py file according to a 3-stage pattern: inference(), loss(), and training().
 <li>inference() - Builds the graph as far as is required for running the network forward to make predictions.</li>
<li>loss() - Adds to the inference graph the ops required to generate loss.</li>
<li>training() - Adds to the loss graph the ops required to compute and apply gradients.</li>

说明在建图的过程中，如果图的结构特别复杂，所有的代码都写在一起就会非常的混乱。所以我们就可以将代码通过函数的形式分开。

## 使用类CLASS
上述的代码组织结构的问题在于
<li>1.很难支持模型复用，也就是面向过程与面向对象的差别</li>
<li>2.没有涉及到数据的处理。因为所有的数据都是使用常用数据集，使用别人写好的api</li>
<font color="red">**所以下面的内容涉及的就是关于使用类重构代码以及写自己的数据集api**</font>
###数据类

###计算model类
[0]:https://www.tensorflow.org/get_started/mnist/mechanics