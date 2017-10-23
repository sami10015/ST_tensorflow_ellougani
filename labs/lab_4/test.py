import tensorflow as tf

from tensorflow.contrib.data import Dataset, Iterator

#Galaxy data
train_imgs = tf.constant(['train/img1.jpg', 'train/img2.jpg', 'train/img4.jpg', 'train/img5.jpg'])

#Desired output, 0 for spiral, 1 for twin
train_labels = tf.constant([0, 0, 1, 1])

#Galaxy test data
test_imgs = tf.constant(['test/img3.jpg', 'test/img6.jpg'])

#Desired test output, 0 for spiral, 1 for twin
test_labels = tf.constant([0, 1])

#Tensorflow dataset object for training data
tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))

#Tensorflow dataset object for testing data
te_data = Dataset.from_tensor_slices((test_imgs, test_labels))

#Tensorflow iterator object
#output_types = data type
#output_shapes = dimensions
iterator = Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)

next_element = iterator.get_next()

#Initialization process
training_init_op = iterator.make_initializer(tr_data)
testing_init_op = iterator.make_initializer(te_data)

with tf.Session() as sess:
	#Initialize the iterator on the training data
	#By running this, you are initializing all the variables associated with tr_data
	sess.run(training_init_op)
	
	while True:
		try:
			elem = sess.run(next_element)
			print(elem)
		except tf.errors.OutOfRangeError:
			print("End of training dataset")
			break
	
	#Initailize the iterator on the test data
	#By running this, you are initializing all the variables associated with te_data
	sess.run(testing_init_op)
	
	while True:
		try:
			elem = sess.run(next_element)
			print(elem)
		except tf.errors.OutOfRangeError:
			print("End of testing dataset")
			break