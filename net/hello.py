import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tfs = tf.compat.v1.InteractiveSession()
hello = tf.constant("Hello TensorFlow !!")
print(tfs.run(hello))

