撰写时间：2017.8.11  
系统环境：ubuntu14.04，tensorflow1.12  
内容：主要介绍tensorflow中name_scope，variable_scope和get_variable函数的使用  
## tf.name_scope()和tf.variable_scope()区别
在tensorflow中，两个函数均是用于定义命名空间。二者的区别主要在于<font color = 'red'>**variable_scope可以与get_variable()函数实现变量的reuse。**</font>下面通过例子讲解
### tf.name_scope函数
```python
import tensorflow as tf
with tf.name_scope("compute") as scope:
    a = tf.Variable(0.1,name="a")
    b = tf.get_variable("b",initializer = 1.0)
    with tf.name_scope("add"):
        c = tf.add(a,b)
print a.name
print b.name
print c.name
"""
result:
compute/a:0
b:0
compute/add/Add:0
"""
```

### tf.variable_scope()函数
```python
with tf.variable_scope("variable_scope"):   
    with tf.name_scope("name_scope") as scope:
        a = tf.Variable(0.1,name="a")
        b = tf.get_variable("b",initializer = 1.0)
        with tf.name_scope("add"):
            c = tf.add(a,b)
print a.name
print b.name
print c.name
"""
result:
variable_scope/name_scope/a:0
variable_scope/b:0
variable_scope/name_scope/add/Add:0
"""
```

从上面的例子可以看出，**name_scope函数会忽略get_variable函数初始化的变量**，即不在变量前加prefix。而variable_scope函数不会忽略。

## tf.Variable()和tf.get_variable()区别
当出现同名的变量时，<font color='red'>Variable()函数会自动进行冲突处理，而get_variables()会raise ValueError exception。</font>下面是相关例子
### tf.Variable()
```python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:09:47 2017

@author: lrh
"""
import tensorflow as tf
with tf.variable_scope("test"):
    a_1 = tf.Variable(0.1,"a")
    a_2 = tf.Variable(0.1,"a")
print a_1.name,a_2.name
"""
result:
test/Variable:0 test/Variable_1:0
"""
```
<font color="red">a_1和a_2是名称是一样的，由于是variable定义的，所以会在变量名称后面添加<i>"_1"</i>作为区分。</font>
### tf.get_variable()
```python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:09:47 2017
@author: lrh
"""
with tf.variable_scope("get_variable"):
    b_1 = tf.get_variable(name="b",initializer=0.1)
    b_2 = tf.get_variable(name="b",initializer=0.1)
print b_1.name,b_2.name
"""
ERROR:
ValueError: Variable get_variable/b already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
"""
```
由于系统中变量出现命名冲突，且get_variable并不会自动处理冲突，所以就会出现ERROR
这样设计的原因是为了方便共享变量的实现

## tf.get_variable()和tf.variable_scope()配合实现共享变量
<font color="red">tf.variable()生成的变量如果想要参数共享，就需要将变量定义成全局的变量。</font>tensorflow使用了get_variable来实现变量共享，variable_scope的命名空间会管理这些共享变量
下面举例子说明几种代码重用的方式
### 不同的scope中共享
```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
print v1.name
"""
result:
foo/v:0
"""

错误实例：
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v1", [1])
"""
ERROR:
ValueError: Variable foo/v1 does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
"""
```
需要注意的一点是：<font color="red">既然scope已经被定义成了reuse属性，那么在之后使用的get_variable的属性全部都要是已经定义好的不然会报错</font>,参考上面错误实例的例子

### 同一个scope中共享
```python
with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v",initializer=1.0)
    scope.reuse_variables()
    #tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v",initializer=1.0)
print v.name,v1.name
```
同上面一样，在设置为reuse之后，该作用于下的之后所有的get_variable所涉及的变量全部都要是共享变量。
关于tensorflow的变量共享可以参看下面的文章：[【极客学院】][0]

<br />
<br />
<font size=5>问题：</font>官方文档中给的cnn的例子，为什么我在自己实现cnn的时候没有注意到共享变量这一点？关于cnn的代码是否需要考虑共享变量？
<font size=5>答案：</font>在我的cnn代码中并没有涉及共享变量这个问题。以为图像的卷积操作op从头到尾也只调用了一次。不会出现[官方文档][1]中所说的  

```python
# First call creates one set of 4 variables.
result1 = my_image_filter(image1)
# Another set of 4 variables is created in the second call.
result2 = my_image_filter(image2)
```

调用两次，<font color ="red">（调用两次的意思是：在计算图中创建两个卷积的操作）</font>所以在我们的cnn代码中不需要考虑共享变量

<font size=5>问题：</font>get_variable函数的initializer参数的可能取值？
<font size=5>答案：</font>该函数常用的传入参数有：name，shape，dtype，initializer，其中initializer的可能取值为：
<li>tf.constant_initializer(value)</li>
<li>tf.random_uniform_initializer(a,b)</li>
<li>tf.random_normal_initializer(mean,stddev)</li>


[0]:http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html  

[1]:https://www.tensorflow.org/programmers_guide/variable_scope
