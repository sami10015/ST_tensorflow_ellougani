import tensorflow as tf

from tensorflow.contrib.data import Dataset, Iterator

#We have two classes
NUM_CLASSES = 2

def image_parser(img_path, label):
	#Convert the label to one-hot encoding
	#TODO: Read up on this
	one_hot = tf.one_hot(label, NUM_CLASSES)
	
	#Read the image from disk
	img_file = tf.read_file(img_path)
	#Channels is 3 for red, green, blue (3 Colors)
	img_decoded = tf.image.decode_image(img_file, channels=3)
	
	return img_decoded, one_hot

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

#Transform the data
tr_data = tr_data.map(image_parser)
te_data = te_data.map(image_parser)

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
