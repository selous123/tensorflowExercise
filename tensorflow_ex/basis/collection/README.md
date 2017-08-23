# collection

tensorflow使用collection管理变量和操作的方式，大大简化了传统编程中关于局部，全局变量的考虑。**例如，即使是在函数中定义的变量variable，只要添加到了TRAINABLE_VARIABLES这个collections中，在session中就会得到更新。**

在编码的过程中，不断地在各处将操作或者变量加入到collections中，然后在run的时候直接取出来用。“零存整取”的思想使得操作异常的简单

## API

```python
tf.add_to_collection(key_name,value)
tf.get_collection(key_name,scope)
"""
Args:
	scope:使用正则表达式，匹配list中的元素。
"""
tf.get_collection_ref(key_name)
```

注：使用的都是默认的Graph。没有考虑多个Graph的情况。以我现在的水平，也还没有到要考虑多个Graph的时候


## standard keys

tf.GraphKeys.TRAINABLE_VALUES：所有的会被optimizer更新的变量
tf.GraphKeys.SUMMARIES：所有的会在tensorboard中显示的summary操作。

上面这是两个到目前为止我用过的standard_keys

其他的官网有[详细介绍](https://www.tensorflow.org/api_docs/python/tf/GraphKeys)
