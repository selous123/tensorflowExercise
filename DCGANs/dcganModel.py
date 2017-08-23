import tensorflow as tf

def lazy_property(func):
    attr_name = "_lazy_" + func.__name__
 
    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
 
    return _lazy_property
class DCGANsModel(object):
    def __init__(self,images,z):
        """
        Args:
            images:shape,[batch_size,height*width]
            z     :shape,[batch_size,100]
        """
        self.images = images
        self.z = z
        self.batch_size = 256
        self.D_theta = []
        self.G_theta = []

    def generator(self,reuse=None):
        
        with tf.variable_scope("g_nn1"):
            weights = tf.get_variable("weights",shape=[100,1024],\
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias",shape=[1024],initializer=tf.constant_initializer(0.1))
            nn1_output = tf.nn.relu(tf.matmul(self.z,weights)+bias)
            self.G_theta.append(weights)
            self.G_theta.append(bias)
        with tf.variable_scope("g_nn2"):
            weights = tf.get_variable("weights",shape=[1024,7*7*128],\
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias",shape=[7*7*128],initializer=tf.constant_initializer(0.1))
            nn2_output = tf.nn.relu(tf.matmul(nn1_output,weights)+bias)
            self.G_theta.append(weights)
            self.G_theta.append(bias)
        with tf.variable_scope("g_reshape"):
            image_input = tf.reshape(nn2_output,shape=[-1,7,7,128],name="reshape")
        
        with tf.variable_scope("g_deconv1"):
            weights = tf.get_variable("weights",shape=[14,14,64,128]\
                                      ,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias",shape=[64],\
                                   initializer = tf.constant_initializer(0.1))
            output_shape=[self.batch_size,14,14,64]
            deconv1 = tf.nn.tanh(tf.nn.conv2d_transpose(image_input,weights\
                                                        ,output_shape=output_shape,padding="SAME",strides=[1,2,2,1])+bias)
            self.G_theta.append(weights)
            self.G_theta.append(bias)
        with tf.variable_scope("g_deconv2"):
            weights = tf.get_variable("weights",shape=[28,28,1,64]\
                                      ,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias",shape=[1],\
                                   initializer = tf.constant_initializer(0.1))
            output_shape=[self.batch_size,28,28,1]
            deconv2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(deconv1,weights,output_shape=output_shape,\
                                             padding="SAME",strides=[1,2,2,1])+bias)
            self.G_theta.append(weights)
            self.G_theta.append(bias)
        
        with tf.variable_scope("g_reshape2"):
            image_output = tf.reshape(deconv2,shape=[-1,28*28],name="reshape")
        
        return image_output
        
    def discriminator(self,z=None,reuse=None):
        if z is None:
            images = self.images
        else:
            images = z
        
        with tf.variable_scope("reshape",reuse=reuse):
            images_input = tf.reshape(images,shape=[-1,28,28,1],name="reshape")
        
        with tf.variable_scope("conv1",reuse=reuse):
            weights = tf.get_variable(name="weights",shape=[5,5,1,32],\
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)) 
            bias = tf.get_variable(name="bias",shape=[32],initializer=tf.constant_initializer(0.1))
            conv1 = tf.nn.relu(tf.nn.conv2d(images_input,weights,strides=[1,1,1,1],padding="SAME")+bias)
            self.D_theta.append(weights)
            self.D_theta.append(bias)
        with tf.variable_scope("pool1",reuse=reuse):
            pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            
        with tf.variable_scope("conv2",reuse=reuse):
            weights = tf.get_variable(name="weights",shape=[5,5,32,64],\
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)) 
            bias = tf.get_variable(name="bias",shape=[64],initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding="SAME")+bias)
            self.D_theta.append(weights)
            self.D_theta.append(bias)
        with tf.variable_scope("pool2",reuse=reuse):
            pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        
        with tf.variable_scope('nn1',reuse=reuse):
            pool2_flat = tf.reshape(pool2,[-1,7*7*64]);
            weight = tf.get_variable(name="weights",shape=[7*7*64,1024],\
                                     initializer = tf.truncated_normal_initializer(stddev = 0.01));
            bias = tf.get_variable(name="bias",shape=[1024],initializer=tf.constant_initializer(0.1));
            nn1_output = tf.nn.tanh(tf.matmul(pool2_flat,weight)+bias);
            self.D_theta.append(weights)
            self.D_theta.append(bias)
            
        with tf.variable_scope('sigmoid',reuse=reuse):
            weight = tf.get_variable(name="weights",shape=[1024,1],\
                                     initializer=tf.truncated_normal_initializer(stddev = 0.01));
            bias = tf.get_variable(name="bias",shape=[1],initializer=tf.constant_initializer(0.1))
            logits = tf.nn.sigmoid(tf.matmul(nn1_output,weight)+bias);
            self.D_theta.append(weights)
            self.D_theta.append(bias)
        return logits
    @lazy_property
    def inference(self):
        """
        build the graph
        """
        g_sample = self.generator()
        D_logits_real = self.discriminator()
        D_logits_fake = self.discriminator(g_sample,reuse=True)
        return g_sample,D_logits_real,D_logits_fake
    @lazy_property
    def optimize(self):
        """
        compute loss
        """
        _,D_logits_real,D_logits_fake = self.inference
        d_loss = -tf.reduce_mean(tf.log(D_logits_real)+tf.log(1. -D_logits_fake))
        g_loss = -tf.reduce_mean(tf.log(D_logits_fake))
        
        tf.summary.scalar("d_loss",d_loss)
        tf.summary.scalar("g_loss",g_loss)
        d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=self.D_theta)
        g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=self.G_theta)
        return d_loss,g_loss,d_optimizer,g_optimizer

    @lazy_property
    def error(self):
        pass
        