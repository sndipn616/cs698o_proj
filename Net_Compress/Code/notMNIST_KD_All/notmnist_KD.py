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

gpu_num = 3

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

current_device = '/gpu:' + str(gpu_num)
export_dir_teacher = 'notMNIST_Model_Teacher/'
export_dir_student = 'notMNIST_Model_Student/'
export_dir_init_student = 'Initial_Wts_Student/'
export_dir = 'notMNIST_Model_Student_KD/'
temp_dir = 'notMNIST_Model_Student/'
# model_saver = tf.saved_model.builder.SavedModelBuilder(export_dir)

model_name = 'notmnist_tf_basic'
model_name_save_teacher = 'notmnist_teacher'
model_name_save_student_trained = 'notmnist_student'
model_name_save_student = 'notmnist_student_KD'
model_name_initial_student = 'notmnist_student_init'
data_dir = 'notMNIST_data/'

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

def test_accuracy(session,teacher=True):

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

  wrong = 0
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

      # feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels, 'x:0' : new_batch_data, 'y:0' : new_batch_labels}

      # if teacher:
      #   feed_dict = {'x:0' : new_batch_data, 'y:0' : new_batch_labels}
      #   [predictions] = session.run([prediction_teacher], feed_dict=feed_dict)
      # else:
      feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels}
      if teacher:
        [predictions] = session.run([prediction_teacher_eval], feed_dict=feed_dict)
      else:
         [predictions] = session.run([prediction_student], feed_dict=feed_dict)

      c, t, w = num_correct_total(predictions, new_batch_labels)
      correct += c
      total += t
      wrong += w

  testImages.close()
  testLabels.close()

  return 100.0 * correct / total, wrong


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

        feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels}
        # feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels, 'x:0' : new_batch_data, 'y:0' : new_batch_labels}
        _, l, predictions = session.run([optimizer_student, loss_student, prediction_student], feed_dict=feed_dict)
        # l, predictions, _, _ = session.run([loss_student, prediction_student, logits_teacher, prediction_teacher], feed_dict=feed_dict)

        if minibatch_num % 10 == 0:
          # print (type(l))
          print('Minibatch loss at step %d and epoch %d : %f' % (minibatch_num, epoch, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))

    images.close()
    labels.close()
        

  model_saver = tf.train.Saver(var_list=student_parameters)
  model_saver.save(session, export_dir + model_name_save_student + str(alpha) + '_' + str(T), write_meta_graph=True)

  acc, w = test_accuracy(session, teacher=False)
  print('Student : Iterations %d, alpha = %f, T = %d, Number of wrong classificiation: %d Test accuracy: %.1f%%' % (num_epochs_student, alpha, T, w, acc))



batch_size = 500
patch_size_teacher = 5
patch_size_student = 5
depth_teacher = 64
depth_student = 16
num_hidden_teacher = 1000
num_hidden_student = 200
# num_epochs_teacher = 3
num_epochs_student = 10
T = 5
prob = 1

alpha = 10
beta = 0.001

# def make_student_graph_KD():
graph_student_KD = tf.Graph()

with graph_student_KD.as_default():
  '''Input data'''
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name='x')
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='y')
  # tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)
  # tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  # tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  '''Variables For Teacher'''
  # Input to Conv1 Layer    
  layer1_weights_teacher = tf.Variable(tf.truncated_normal([patch_size_teacher, patch_size_teacher, num_channels, depth_teacher], stddev=0.1), name='l1wt')
  layer1_biases_teacher = tf.Variable(tf.zeros([depth_teacher]), name='l1bt')

  # Conv1 to Conv2 Layer    
  layer2_weights_teacher = tf.Variable(tf.truncated_normal([patch_size_teacher, patch_size_teacher, depth_teacher, depth_teacher], stddev=0.1), name='l2wt')
  layer2_biases_teacher = tf.Variable(tf.zeros([depth_teacher]), name='l2bt')
  
  teacher_first_half_params = [layer1_weights_teacher, layer1_biases_teacher, layer2_weights_teacher, layer2_biases_teacher]
  # Conv2 to FC1 Layer
  final_image_size = 7
  layer3_weights_teacher = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth_teacher, num_hidden_teacher], stddev=0.1), name='l3wt')
  layer3_biases_teacher = tf.Variable(tf.constant(1.0, shape=[num_hidden_teacher]), name='l3bt')

  # FC1 to FC2 Layer
  layer4_weights_teacher = tf.Variable(tf.truncated_normal([num_hidden_teacher, num_labels], stddev=0.1), name='l4wt')
  layer4_biases_teacher = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='l4bt')

  teacher_second_half_params = [layer3_weights_teacher, layer3_biases_teacher, layer4_weights_teacher, layer4_biases_teacher]

  teacher_parameters = [layer1_weights_teacher, layer1_biases_teacher, layer2_weights_teacher, layer2_biases_teacher, layer3_weights_teacher, layer3_biases_teacher, layer4_weights_teacher, layer4_biases_teacher]
  # data = tf_train_dataset
  '''Teacher Model for Training'''
  def teacher_model_train(data):
    # First Convolutional Layer with Pooling
    conv_1 = tf.nn.conv2d(data, layer1_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
    hidden_1 = tf.nn.relu(conv_1 + layer1_biases_teacher)
    pool_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Second Convolutional Layer with Pooling
    conv_2 = tf.nn.conv2d(pool_1, layer2_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
    hidden_2 = tf.nn.relu(conv_2 + layer2_biases_teacher)
    pool_2 = tf.nn.max_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Full Connected Layer
    # print ("Shape - ")
    # print (pool_2.get_shape())
    shape = pool_2.get_shape().as_list()
    reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights_teacher) + layer3_biases_teacher)
    
    # Readout Layer: Softmax Layer
    return tf.matmul(hidden, layer4_weights_teacher) + layer4_biases_teacher

  def teacher_model_eval(data):
    # First Convolutional Layer with Pooling
    conv_1 = tf.nn.conv2d(data, layer1_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
    hidden_1 = tf.nn.relu(conv_1 + layer1_biases_teacher)
    pool_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Second Convolutional Layer with Pooling
    conv_2 = tf.nn.conv2d(pool_1, layer2_weights_teacher, strides=[1, 1, 1, 1], padding='SAME')
    hidden_2 = tf.nn.relu(conv_2 + layer2_biases_teacher)
    pool_2 = tf.nn.max_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Full Connected Layer
    shape = pool_2.get_shape().as_list()
    reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights_teacher) + layer3_biases_teacher)
    
    # Readout Layer: Softmax Layer
    return tf.matmul(hidden, layer4_weights_teacher) + layer4_biases_teacher
     
  # logits = tf.matmul(hidden, layersm_weights_teacher) + layersm_biases_teacher
  '''Training computation'''   
  logits_teacher_eval = teacher_model_eval(tf_train_dataset)

  tf.add_to_collection("teacher_model_logits", logits_teacher_eval)
  
  prediction_teacher_eval = tf.nn.softmax(logits_teacher_eval)

  tf.add_to_collection("teacher_model_prediction", prediction_teacher_eval)

  '''Variables For Student'''
  # Input to Conv1 Layer    
  layer1_weights_student = tf.Variable(tf.truncated_normal([patch_size_student, patch_size_student, num_channels, depth_student], stddev=0.1), name='l1ws')
  layer1_biases_student = tf.Variable(tf.zeros([depth_student]), name='l1bs')
  
  # Conv1 to Conv2 Layer 
  layer2_weights_student = tf.Variable(tf.truncated_normal([patch_size_student, patch_size_student, depth_student, depth_student], stddev=0.1), name='l2ws')
  layer2_biases_student = tf.Variable(tf.constant(1.0, shape=[depth_student]), name='l2bs')

  # Conv2 to Conv3 Layer 
  layer3_weights_student = tf.Variable(tf.truncated_normal([patch_size_student, patch_size_student, depth_student, depth_student], stddev=0.1), name='l3ws')
  layer3_biases_student = tf.Variable(tf.constant(1.0, shape=[depth_student]), name='l3bs')

  student_first_half_params = [layer1_weights_student, layer1_biases_student, layer2_weights_student, layer2_biases_student, layer3_weights_student, layer3_biases_student]
  # Conv3 to FC1 Layer
  final_image_size = 4
  layer4_weights_student = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth_student, num_hidden_student], stddev=0.1), name='l4ws')
  layer4_biases_student = tf.Variable(tf.constant(1.0, shape=[num_hidden_student]), name='l4bs')

  # FC1 to FC2 Layer
  layer5_weights_student = tf.Variable(tf.truncated_normal([num_hidden_student, num_hidden_student], stddev=0.1), name='l5ws')
  layer5_biases_student = tf.Variable(tf.constant(1.0, shape=[num_hidden_student]), name='l5bs')

  # FC2 to FC3 Layer
  layer6_weights_student = tf.Variable(tf.truncated_normal([num_hidden_student, num_labels], stddev=0.1), name='l6ws')
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
  logits_student_soft = logits_student / T

  prediction_teacher_soft = tf.nn.softmax(logits_teacher_eval / T)
  

  tf.add_to_collection("student_model_logits", logits_student)
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
  loss_student = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_student)) \
  + alpha*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=prediction_teacher_soft, logits=logits_student_soft)))

  '''Optimizer'''
  # Learning rate of 0.05
  optimizer_student = tf.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(loss_student, var_list=student_parameters)
  # optimizer_student = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_student) 

  '''Predictions for the training, validation, and test data'''
  prediction_student = tf.nn.softmax(logits_student)
  
  tf.add_to_collection("student_model_prediction", prediction_student)


def train_student_KD():
  with tf.device(current_device):
    # graph_student_KD = make_student_graph_KD()

    with tf.Session(graph=graph_student_KD) as session:
      tf.global_variables_initializer().run()
      
      saver = tf.train.Saver(var_list=teacher_parameters)
      saver.restore(session, export_dir_teacher + model_name_save_teacher)

      try:
        saver = tf.train.Saver(var_list=student_parameters)
        saver.restore(session, export_dir_init_student + model_name_initial_student)
      except:
        pass

      print ("Testing Teacher for sanity check")
      acc, w = test_accuracy(session)
      print('Teacher : Number of wrong classificiation: %d Test accuracy: %.1f%%' % (w, acc))

      Train_Student(session)


train_student_KD()