#By Sami Ellougani
import numpy as np
import tensorflow as tf

#Multiplying matrix's with numpy
a = np.array([[1, 2], [4, -1], [-3, 3]])
b = np.array([[-2, 0, 5], [0, -1, 4]])
print(np.dot(a,b))

#Multiplying matrix's with tensorflow
a = tf.constant(a)
b = tf.constant(b)

with tf.Session() as sess:
	print(tf.matmul(a,b).eval())