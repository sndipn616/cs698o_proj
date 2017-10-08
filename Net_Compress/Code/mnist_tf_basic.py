from __future__ import print_function
import gzip
import numpy as np
from struct import unpack
# from skimage.feature import hog
# from skimage import data, color
from numpy import zeros, uint8
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", validation_size=10000, one_hot=True)
current_device = '/cpu:0'
export_dir = 'MNIST_Models/'
# model_saver = tf.saved_model.builder.SavedModelBuilder(export_dir)

model_name = 'mnist_tf_basic'
data_dir = 'MNIST_data/'

images = gzip.open(data_dir + "train-images-idx3-ubyte.gz", 'rb')
labels = gzip.open(data_dir + "train-labels-idx1-ubyte.gz", 'rb')

images.read(4)
NumImages = images.read(4)
NumImages = unpack('>I', NumImages)[0]
# NumImages = 10000
NumRows = images.read(4)
NumRows = unpack('>I', NumRows)[0]
NumColumns = images.read(4)
NumColumns = unpack('>I', NumColumns)[0]

labels.read(4)  
NumLabels = labels.read(4)
NumLabels = unpack('>I', NumLabels)[0]

testImages = gzip.open(data_dir + "t10k-images-idx3-ubyte.gz", 'rb')
testLabels = gzip.open(data_dir + "t10k-labels-idx1-ubyte.gz", 'rb')

testImages.read(4)
NumImages2 = testImages.read(4)
NumImages2 = unpack('>I', NumImages2)[0]

NumRows2 = testImages.read(4)
NumRows2 = unpack('>I', NumRows2)[0]
NumColumns2 = testImages.read(4)
NumColumns2 = unpack('>I', NumColumns2)[0]

testLabels.read(4)  
NumLabels2 = testLabels.read(4)
NumLabels2 = unpack('>I', NumLabels2)[0]

# pickle_file = 'notMNIST.pickle'

# with open(pickle_file, 'rb') as f:
#   save = pickle.load(f)
#   train_dataset = save['train_dataset']
#   train_labels = save['train_labels']
#   valid_dataset = save['valid_dataset']
#   valid_labels = save['valid_labels']
#   test_dataset = save['test_dataset']
#   test_labels = save['test_labels']
#   del save  # hint to help gc free up memory
#   print('Training set', train_dataset.shape, train_labels.shape)
#   print('Validation set', valid_dataset.shape, valid_labels.shape)
#   print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

image_size = 28
num_labels = 10
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  # labels = np.arange(num_labels) == labels[:,None]

  return dataset, labels

train_dataset, train_labels = reformat(mnist.train.images, mnist.train.labels)
# valid_dataset, valid_labels = reformat(mnist.validation.images, mnist.validation.labels)
# test_dataset, test_labels = reformat(mnist.test.images, mnist.test.labels)

# del mnist
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def output_size_no_pool(input_size, filter_size, padding, conv_stride):
    if padding == 'same':
        padding = -1.00
    elif padding == 'valid':
        padding = 0.00
    else:
        return None
    output_1 = float(((input_size - filter_size - 2*padding) / conv_stride) + 1.00)
    output_2 = float(((output_1 - filter_size - 2*padding) / conv_stride) + 1.00)
    return int(np.ceil(output_2))


batch_size = 10
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

    '''Input data'''
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    # tf_valid_dataset = tf.constant(valid_dataset)
    # tf_test_dataset = tf.constant(test_dataset)
    # tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    # tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    '''Variables'''
    # Convolution 1 Layer
    # Input channels: num_channels = 1
    # Output channels: depth = 16
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    
    # Convolution 2 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    # Convolution 3 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    
    # Fully Connected Layer (Densely Connected Layer)
    # Use neurons to allow processing of entire image
    # final_image_size = output_size_no_pool(image_size, patch_size, padding='same', conv_stride=2)
    final_image_size = 4
    layer4_weights = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1))    
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    # Readout layer: Softmax Layer
    layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    data = tf_train_dataset
    '''Model'''
    # def teacher_model(data):
      # First Convolutional Layer with Pooling
    conv_1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
    pool_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Second Convolutional Layer with Pooling
    conv_2 = tf.nn.conv2d(pool_1, layer2_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
    pool_2 = tf.nn.max_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # print ("Pool2")
    # print (pool_2.get_shape())
    # Third Convolutional Layer with Pooling
    conv_3 = tf.nn.conv2d(pool_2, layer3_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden_3 = tf.nn.relu(conv_3 + layer3_biases)
    pool_3 = tf.nn.max_pool(hidden_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # print ("Pool3")
    # print (pool_3.get_shape())
    
    # Full Connected Layer
    shape = pool_3.get_shape().as_list()
    reshape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
        
        # Readout Layer: Softmax Layer
        # return tf.matmul(hidden, layer5_weights) + layer5_biases
    logits = tf.matmul(hidden, layer5_weights) + layer5_biases
    '''Training computation'''
    # logits = teacher_model(tf_train_dataset)

#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    '''Optimizer'''
    # Learning rate of 0.05
    optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

    '''Predictions for the training, validation, and test data'''
    prediction = tf.nn.softmax(logits)
    # valid_prediction = tf.nn.softmax(teacher_model(tf_valid_dataset))
    # test_prediction = tf.nn.softmax(teacher_model(tf_test_dataset))


def test_accuracy(session):
  for i in range(NumLabels):      
    CurrImage = np.zeros((NumRows,NumColumns), dtype=np.float32)
    for row in range(NumRows2):
      for col in range(NumColumns2):
        pixelValue = images.read(1)  
        pixelValue = unpack('>B', pixelValue)[0]
        # print (pixelValue)
        CurrImage[row][col] = pixelValue * 1.0
        
        
    batch_data.append(CurrImage)
    # print (CurrImage)
    labelValue = labels.read(1)      
    labelValue = unpack('>B', labelValue)[0]
    batch_labels.append(labelValue)

    count_in_batch += 1
    if count_in_batch >= batch_size:
      count_in_batch = 0
      # CurrImage = np.zeros((batch_size,NumColumns), dtype=uint8)
      minibatch_num += 1

      batch_data = np.array(batch_data)
      batch_labels = np.array(batch_labels)

      new_batch_data, new_batch_labels = reformat(batch_data, batch_labels)

      # print (new_batch_labels)
      # print (new_batch_data.shape)
      # print (new_batch_labels.shape)

      batch_data = []
      batch_labels = []



with tf.device(current_device):
#   # num_steps = 10000
#   # if 'gpu' not in current_device:
#   #   num_steps = 5000

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    batch_data = []
    batch_labels = []
    count_in_batch = 0
    minibatch_num = 0
    for i in range(NumLabels):      
      CurrImage = np.zeros((NumRows,NumColumns), dtype=np.float32)
      for row in range(NumRows):
        for col in range(NumColumns):
          pixelValue = images.read(1)  
          pixelValue = unpack('>B', pixelValue)[0]
          # print (pixelValue)
          CurrImage[row][col] = pixelValue * 1.0
          
          
      batch_data.append(CurrImage)
      # print (CurrImage)
      labelValue = labels.read(1)      
      labelValue = unpack('>B', labelValue)[0]
      batch_labels.append(labelValue)

      count_in_batch += 1
      if count_in_batch >= batch_size:
        count_in_batch = 0
        # CurrImage = np.zeros((batch_size,NumColumns), dtype=uint8)
        minibatch_num += 1

        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        new_batch_data, new_batch_labels = reformat(batch_data, batch_labels)

        # print (new_batch_labels)
        # print (new_batch_data.shape)
        # print (new_batch_labels.shape)

        batch_data = []
        batch_labels = []

        feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels}
        _, l, predictions, log, c = session.run([optimizer, loss, prediction, logits, layer1_weights], feed_dict=feed_dict)

        if minibatch_num % 100 == 0:
          print('Minibatch loss at step %d: %f' % (minibatch_num, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))
          

    acc = test_accuracy(session)
    print('Test accuracy: %.1f%%' % acc)



  # num_steps = 10000
  # if 'gpu' not in current_device:
  #   num_steps = 5000

  # with tf.Session(graph=graph) as session:
  #   tf.global_variables_initializer().run()
  #   # tf.saved_model.loader.load(session, [tag_constants.TRAINING], export_dir)
  #   print('Initialized')
  #   for step in range(num_steps):
  #     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
  #     batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
  #     batch_labels = train_labels[offset:(offset + batch_size), :]
  #     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
  #     _, l, predictions = session.run(
  #       [optimizer, loss, train_prediction], feed_dict=feed_dict)
  #     if (step % 100 == 0):
  #       print('Minibatch loss at step %d: %f' % (step, l))
  #       print (predictions)
  #       print (batch_labels)
  #       print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        # print('Validation accuracy: %.1f%%' % accuracy(
        #   valid_prediction.eval(), valid_labels))

  #   model_saver = tf.train.Saver()
  #   model_saver.save(session, export_dir + model_name, write_meta_graph=False)

  #   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    # model_saver.add_meta_graph_and_variables(session,[tag_constants.TRAINING])

  # with tf.Session(graph=graph) as session:
  #   tf.global_variables_initializer().run()
  #   print('Second Phase Training')
  #   # saver = tf.train.import_meta_graph(export_dir + 'mnist_tf_basic-4900.meta')
  #   saver = tf.train.Saver()
  #   saver.restore(session, tf.train.latest_checkpoint(export_dir))
  #   for step in range(num_steps):
  #     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
  #     batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
  #     batch_labels = train_labels[offset:(offset + batch_size), :]
  #     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
  #     _, l, predictions = session.run(
  #       [optimizer, loss, train_prediction], feed_dict=feed_dict)
  #     if (step % 100 == 0):
  #       print('Minibatch loss at step %d: %f' % (step, l))
  #       print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
  #       print('Validation accuracy: %.1f%%' % accuracy(
  #         valid_prediction.eval(), valid_labels))

  #   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    

