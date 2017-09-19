from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

#Weight tensor
W = tf.Variable(tf.zeros([784,10], tf.float32))
#Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))

#TensorFlow needs to initialize the variables that you assign
#Run the op initialize_all_variables using an interactive session
sess.run(tf.initialize_all_variables())

#Mathematical operation to add weights and biases to the inputs
tf.matmul(x,W) + b

#Softmax is an activation function that is normally used in classification problems
#It generates the probabilities for the output
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Cost function
#A function that is used to minimize the difference between right answers and estimated outputs
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices[1]))

#Type of optimization: Gradient Descent
#Configure optimizer for Neural Network
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Training batches
#Load 50 training examples for each training iteration
for i in range(1000):
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

sess.close()