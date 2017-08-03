import tensorflow as tf


##build graph;
weights = tf.Variable(tf.ones([10,3]),tf.float32);
#bias = tf.Variable(tf.ones([1,3]),tf.float32);

bias = tf.constant([[1],[1],[1],[1],[1]],tf.float32);
x = tf.Variable(tf.ones([5,10]),tf.float32)


y_ = tf.matmul(x,weights)+bias;

sess = tf.Session()
init = tf.global_variables_initializer();
sess.run(init);

print bias.eval(session = sess);
print sess.run(y_)