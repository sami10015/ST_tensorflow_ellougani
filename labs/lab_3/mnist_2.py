import tensorflow as tensorflow

#Start interactive session
sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

width = 28 #Width of the image in pixels
height = 28 #Height of the image in pixels
flat = width * height #Number of pixels in one image
class_output = 10 #Numper of possible classifications for the problem

x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

#Converting images of the data set to tensors
#The input image is 28x28 pixels, 1 channel(grayscale), first parameter is the batch number
#It can be of any size, so we set it to -1
x_image = tf.reshape(x, [-1,28,28,1])
x_image

#Convolutional Layer 1
#The size of the filter/kernel is 5x5; Input channels is 1(greyscale);
#We need 32 different feature maps(here, 32 feature maps means 32 filters applied on each image)
w_conv1 = tf.Variable(tf.truncated([5,5,1,32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) #Need 32 biases for 32 outputs

#Convolve with weight tensor and add biases
#Inputs: tensor of shape, a filter/kernel, stride of kernel window
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

#Apply the ReLU Activation Function
h_conv1 = tf.nn.relu(convolve1)

#Apply the max pooling function(Takes the max of a matrix)
conv2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

#Convolutional Layer 2
#The input channels is 32, because we had 32 feature maps
#64 output feature maps
w_conv2 = tf.Variable(tf.truncated_normal([5,5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #Need 64 biases for 64 outputs

#Convolve image with weight tensors and add biases
convolve2 = tf.nn.conv2d(conv1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2

#Apply the ReLu activation function
h_conv2 = tf.nn.relu(convolve2)

#Apply the max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
