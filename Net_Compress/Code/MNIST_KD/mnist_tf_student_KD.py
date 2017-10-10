from __future__ import print_function
import os
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

# mnist = input_data.read_data_sets("MNIST_data/", validation_size=10000, one_hot=True)
current_device = '/cpu:0'
export_dir = 'MNIST_Model_Student_KD/'
temp_dir = 'MNIST_Models/'
# model_saver = tf.saved_model.builder.SavedModelBuilder(export_dir)

model_name = 'mnist_tf_basic'
model_name_save_teacher = 'mnist_tf_teacher'
model_name_save_student = 'mnist_tf_student_KD'
data_dir = 'MNIST_data/'

def return_pointers():
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

  return images, labels, NumLabels, NumRows, NumColumns

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

# train_dataset, train_labels = reformat(mnist.train.images, mnist.train.labels)
# valid_dataset, valid_labels = reformat(mnist.validation.images, mnist.validation.labels)
# test_dataset, test_labels = reformat(mnist.test.images, mnist.test.labels)

# del mnist
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def num_correct_total(predictions, labels):
  return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)), predictions.shape[0]


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

def test_accuracy(session):
  correct = 0
  total = 0

  batch_data = []
  batch_labels = []
  count_in_batch = 0
  minibatch_num = 0
  for i in range(NumLabels2):      
    CurrImage = np.zeros((NumRows2,NumColumns2), dtype=np.float32)
    for row in range(NumRows2):
      for col in range(NumColumns2):
        pixelValue = testImages.read(1)  
        pixelValue = unpack('>B', pixelValue)[0]
        # print (pixelValue)
        CurrImage[row][col] = pixelValue * 1.0
        
        
    batch_data.append(CurrImage)
    # print (CurrImage)
    labelValue = testLabels.read(1)      
    labelValue = unpack('>B', labelValue)[0]
    batch_labels.append(labelValue)

    count_in_batch += 1
    if count_in_batch >= batch_size:
      count_in_batch = 0
      
      minibatch_num += 1

      batch_data = np.array(batch_data)
      batch_labels = np.array(batch_labels)

      new_batch_data, new_batch_labels = reformat(batch_data, batch_labels)

      batch_data = []
      batch_labels = []

      feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels}
      [predictions] = session.run([prediction], feed_dict=feed_dict)

      c, t = num_correct_total(predictions, new_batch_labels)
      correct += c
      total += t

  return 100.0 * correct / total

def Train_Teacher(session):  
  for epoch in range(num_epochs_teacher):
    batch_data = []
    batch_labels = []
    count_in_batch = 0
    minibatch_num = 0
    images, labels, NumLabels, NumRows, NumColumns = return_pointers()
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

        batch_data = []
        batch_labels = []

        feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels, tf_flag : True}
        # _, l, predictions = session.run([optimizer_teacher, loss_teacher, prediction_teacher_train], feed_dict=feed_dict)
        l, predictions = session.run([loss, pred], feed_dict=feed_dict)

        if minibatch_num % 100 == 0:
          print('Minibatch loss at step %d: %f' % (minibatch_num, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))

    images.close()
    labels.close()
        

  model_saver = tf.train.Saver()
  model_saver.save(session, export_dir + model_name_save_teacher, write_meta_graph=True)

  acc = test_accuracy(session)
  print('Test accuracy for Teacher: %.1f%%' % acc)

def Train_Student(session):
  for epoch in range(num_epochs_student):
    batch_data = []
    batch_labels = []
    count_in_batch = 0
    minibatch_num = 0
    images, labels, NumLabels, NumRows, NumColumns = return_pointers()
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

        batch_data = []
        batch_labels = []

        feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels, tf_flag : False}
        # _, l, predictions = session.run([optimizer_student, loss_student, prediction_student], feed_dict=feed_dict)
        l, predictions = session.run([loss, pred], feed_dict=feed_dict)

        if minibatch_num % 100 == 0:
          print('Minibatch loss at step %d: %f' % (minibatch_num, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))

    images.close()
    labels.close()
        

  model_saver = tf.train.Saver()
  model_saver.save(session, export_dir + model_name_save_student, write_meta_graph=True)

  acc = test_accuracy(session)
  print('Test accuracy for Student: %.1f%%' % acc)



batch_size = 10
patch_size = 3
depth = 32
num_hidden = 64
num_epochs_teacher = 3
num_epochs_student = 3
alpha = 0.005
T = 100
alpha = 0.5

graph_student_KD = tf.Graph()

with graph_student_KD.as_default():

    '''Input data'''
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_flag = tf.placeholder(tf.bool, shape=[])
    tf_flag_value = tf.constant([1])
    # tf_valid_dataset = tf.constant(valid_dataset)
    # tf_test_dataset = tf.constant(test_dataset)
    # tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    # tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    '''Variables For Teacher'''
    # Convolution 1 Layer
    # Input channels: num_channels = 1
    # Output channels: depth = 16
    layer1_weights_teacher = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases_teacher = tf.Variable(tf.zeros([depth]))
    
    # Convolution 2 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer2_weights_teacher = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases_teacher = tf.Variable(tf.constant(1.0, shape=[depth]))

    # Convolution 3 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer3_weights_teacher = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer3_biases_teacher = tf.Variable(tf.constant(1.0, shape=[depth]))

    # Convolution 4 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer4_weights_teacher = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer4_biases_teacher = tf.Variable(tf.constant(1.0, shape=[depth]))
    
    # Fully Connected Layer (Densely Connected Layer)
    # Use neurons to allow processing of entire image
    # final_image_size = output_size_no_pool(image_size, patch_size, padding='same', conv_stride=2)
    final_image_size = 28
    layerfc_weights_teacher = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1))    
    layerfc_biases_teacher = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    # Readout layer: Softmax Layer
    layersm_weights_teacher = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layersm_biases_teacher = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    teacher_parameters = [layer1_weights_teacher, layer1_biases_teacher, layer2_weights_teacher, layer2_biases_teacher, layer3_weights_teacher, layer3_biases_teacher, layer4_weights_teacher, layer4_biases_teacher, layerfc_weights_teacher, layerfc_biases_teacher, layersm_weights_teacher, layersm_biases_teacher]

    '''Variables For Student'''
    # Convolution 1 Layer
    # Input channels: num_channels = 1
    # Output channels: depth = 16
    layer1_weights_student = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases_student = tf.Variable(tf.zeros([depth]))
    
    
    # Fully Connected Layer (Densely Connected Layer)
    # Use neurons to allow processing of entire image
    # final_image_size = output_size_no_pool(image_size, patch_size, padding='same', conv_stride=2)
    final_image_size = 28
    layerfc_weights_student = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1))    
    layerfc_biases_student = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    # Readout layer: Softmax Layer
    layersm_weights_student = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layersm_biases_student = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    student_parameters = [layer1_weights_student, layer1_biases_student, layerfc_weights_student, layerfc_biases_student, layersm_weights_student, layersm_biases_student]

    # data = tf_train_dataset
    '''Teacher Model'''
    def teacher_model(data,train=True):
        # First Convolutional Layer with Pooling
      conv_1 = tf.nn.conv2d(data, layer1_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
      hidden_1 = tf.nn.relu(conv_1 + layer1_biases_teacher)
      pool_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
      
      # Second Convolutional Layer with Pooling
      conv_2 = tf.nn.conv2d(pool_1, layer2_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
      hidden_2 = tf.nn.relu(conv_2 + layer2_biases_teacher)
      pool_2 = tf.nn.max_pool(hidden_2, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

      # print ("Pool2")
      # print (pool_2.get_shape())
      # Third Convolutional Layer with Pooling
      conv_3 = tf.nn.conv2d(pool_2, layer3_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
      hidden_3 = tf.nn.relu(conv_3 + layer3_biases_teacher)
      pool_3 = tf.nn.max_pool(hidden_3, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

      # Fourth Convolutional Layer with Pooling
      conv_4 = tf.nn.conv2d(pool_2, layer3_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
      hidden_4 = tf.nn.relu(conv_3 + layer3_biases_teacher)
      pool_4 = tf.nn.max_pool(hidden_3, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
     
      shape = pool_4.get_shape().as_list()
      reshape = tf.reshape(pool_4, [shape[0], shape[1] * shape[2] * shape[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, layerfc_weights_teacher) + layerfc_biases_teacher)
        
        # Readout Layer: Softmax Layer
      return tf.matmul(hidden, layersm_weights_teacher) + layersm_biases_teacher
    # logits = tf.matmul(hidden, layersm_weights_teacher) + layersm_biases_teacher
    '''Training computation'''
    def train_teacher():
      print ("Teacher")
      logits_teacher = teacher_model(tf_train_dataset)
      loss_teacher = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_teacher)) 
      '''Optimizer'''
      # Learning rate of 0.05
      optimizer_teacher = tf.train.GradientDescentOptimizer(0.001).minimize(loss_teacher, var_list=teacher_parameters)

      '''Predictions for the training, validation, and test data'''
      prediction_teacher_train = tf.nn.softmax(logits_teacher)

      return loss_teacher, prediction_teacher_train

    # if tf_flag is tf_flag_value :
      # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))      


    def student_model(data,train=True):
      # First Convolutional Layer with Pooling
      conv_1 = tf.nn.conv2d(data, layer1_weights_student, strides=[1, 1, 1, 1], padding='SAME')
      hidden_1 = tf.nn.relu(conv_1 + layer1_biases_student)
      pool_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
      
      shape = pool_1.get_shape().as_list()
      reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, layerfc_weights_student) + layerfc_biases_student)
        
        # Readout Layer: Softmax Layer
      return tf.matmul(hidden, layersm_weights_student) + layersm_biases_student


    def train_student():
      print ("Student")
      logits_teacher = teacher_model(tf_train_dataset)
      logits_student = student_model(tf_train_dataset)    

      # prediction_student_soft = tf.nn.softmax(logits_student / T)
      prediction_teacher_soft = tf.nn.softmax(logits_teacher / T)
      # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
      loss_student = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_student)) \
      + alpha*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=prediction_teacher_soft, logits=logits_student / T)))

      '''Optimizer'''
      # Learning rate of 0.05
      optimizer_student = tf.train.GradientDescentOptimizer(0.001).minimize(loss_student, var_list=student_parameters)

      '''Predictions for the training, validation, and test data'''
      prediction_student = tf.nn.softmax(logits_student)

      return loss_student, prediction_student

    loss, pred = tf.cond(tf_flag, train_teacher, train_student)
    # loss, pred = fn(tf_train_dataset, tf_train_labels)



with tf.device(current_device):  
  with tf.Session(graph=graph_student_KD) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    if os.path.isfile(export_dir + model_name_save_teacher + '.meta'):
      saver = tf.train.Saver()
      saver.restore(session, export_dir + model_name_save_teacher)

    print ("Training Teacher")
    Train_Teacher(session)

    print ("Training Student")
    Train_Student(session)


  
    


  
