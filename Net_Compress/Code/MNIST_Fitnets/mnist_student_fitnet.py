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

gpu_num = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

current_device = '/gpu:' + str(gpu_num)
export_dir = 'MNIST_Model_Student/'
temp_dir = 'MNIST_Models/'
# model_saver = tf.saved_model.builder.SavedModelBuilder(export_dir)

model_name_save_student = 'mnist_student_fitnet'
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
  temp = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
  return temp, predictions.shape[0], predictions.shape[0] - temp

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

  correct = 0
  total = 0
  wrong = 0

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
      [predictions] = session.run([prediction_student], feed_dict=feed_dict)

      c, t, w = num_correct_total(predictions, new_batch_labels)
      correct += c
      total += t
      wrong += w

  testImages.close()
  testLabels.close()

  return 100.0 * correct / total, wrong

def Train_Student(session):  
  for epoch in range(num_epochs):
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

        feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels}
        _, l, predictions = session.run([optimizer_student, loss_student, prediction_student], feed_dict=feed_dict)

        if minibatch_num % 10 == 0:
          print('Minibatch loss at step %d: %f' % (minibatch_num, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))

    images.close()
    labels.close()
        

  model_saver = tf.train.Saver(var_list=student_parameters)
  model_saver.save(session, export_dir + model_name_save_student, write_meta_graph=True)

  acc, w = test_accuracy(session)
  print('Student : Total Iteration %d, Number of wrong classificiation: %d Test accuracy: %.1f%%' % (num_epochs, w, acc))
  with open("output.txt", "a") as myfile:
    myfile.write('Student : Total Iteration %d, Number of wrong classificiation: %d Test accuracy: %.1f%% \n' % (num_epochs, w, acc))



batch_size = 500
patch_size = 3
depth = 16


prob = 0.5
num_hidden = 200

num_epochs = 20
beta = 0.001

graph_student = tf.Graph()

with graph_student.as_default():

  '''Input data'''
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name='x')
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='y')
  # tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)
  # tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  # tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  '''Variables For Student'''
  # Input to Conv1 Layer    
  layer1_weights_student = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='l1ws')
  layer1_biases_student = tf.Variable(tf.zeros([depth]), name='l1bs')
  
  # Conv1 to Conv2 Layer 
  layer2_weights_student = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='l2ws')
  layer2_biases_student = tf.Variable(tf.constant(1.0, shape=[depth]), name='l2bs')

  # Conv2 to Conv3 Layer 
  layer3_weights_student = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='l3ws')
  layer3_biases_student = tf.Variable(tf.constant(1.0, shape=[depth]), name='l3bs')

  student_first_half_params = [layer1_weights_student, layer1_biases_student, layer2_weights_student, layer2_biases_student, layer3_weights_student, layer3_biases_student]
  # Conv3 to FC1 Layer
  final_image_size = 4
  layer4_weights_student = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1), name='l4ws')
  layer4_biases_student = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='l4bs')

  # FC1 to FC2 Layer
  layer5_weights_student = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1), name='l5ws')
  layer5_biases_student = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='l5bs')

  # FC2 to FC3 Layer
  layer6_weights_student = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='l6ws')
  layer6_biases_student = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='l6bs')

  student_second_half_params = [layer4_weights_student, layer4_biases_student, layer5_weights_student, layer5_biases_student, layer6_weights_student, layer6_biases_student]

  student_parameters = [layer1_weights_student, layer1_biases_student, layer2_weights_student, layer2_biases_student, layer3_weights_student, layer3_biases_student, layer4_weights_student, layer4_biases_student, layer5_weights_student, layer5_biases_student, layer6_weights_student, layer6_biases_student]


  def student_model(data):
    # First Convolutional Layer with Pooling
    conv_1 = tf.nn.conv2d(data, layer1_weights_student, strides=[1, 1, 1, 1], padding='SAME')
    hidden_1 = tf.nn.relu(conv_1 + layer1_biases_student)
    pool_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Second Convolutional Layer with Pooling
    conv_2 = tf.nn.conv2d(pool_1, layer2_weights_student, strides=[1, 1, 1, 1], padding='SAME')
    hidden_2 = tf.nn.relu(conv_2 + layer2_biases_student)
    pool_2 = tf.nn.max_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # Third Convolutional Layer with Pooling
    conv_3 = tf.nn.conv2d(pool_2, layer3_weights_student, strides=[1, 1, 1, 1], padding='SAME')
    hidden_3 = tf.nn.relu(conv_3 + layer3_biases_student)
    pool_3 = tf.nn.max_pool(hidden_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Full Connected Layer
    # print ("Shape - ")
    # print (pool_3.get_shape())
    shape = pool_3.get_shape().as_list()
    reshape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights_student) + layer4_biases_student)

    hidden = tf.nn.relu(tf.matmul(hidden, layer5_weights_student) + layer5_biases_student)
    
    # Readout Layer: Softmax Layer
    return tf.matmul(hidden, layer6_weights_student) + layer6_biases_student

  # logits = tf.matmul(hidden, layersm_weights_teacher) + layersm_biases_teacher
  '''Training computation'''   
  logits_student = student_model(tf_train_dataset)
  

  tf.add_to_collection("student_model_logits", logits_student)
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
  loss_student = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_student)) 

  '''Optimizer'''
  # Learning rate of 0.05
  optimizer_student = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss_student)
  # optimizer_student = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_student) 

  '''Predictions for the training, validation, and test data'''
  prediction_student = tf.nn.softmax(logits_student)
  
  tf.add_to_collection("student_model_prediction", prediction_student)

  # return graph_student

def pre_train_student():
  with tf.device(current_device): 
    # graph_student = make_student_graph() 
    with tf.Session(graph=graph_student, config=config) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      if os.path.isfile(export_dir + model_name_save_student + '.meta'):
        saver = tf.train.Saver(var_list=student_parameters)
        saver.restore(session, export_dir + model_name_save_student)

      Train_Student(session)  

# pre_train_student()   


  
