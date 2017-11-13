from __future__ import print_function
import os
import gzip
import random
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
export_dir_teacher = 'MNIST_Model_Teacher/'
export_dir_student = 'MNIST_Model_Student/'
export_dir = 'Disc_Model/'
dir_disc_data = 'Disc_data/'
temp_dir = 'MNIST_Model_Student/'
# model_saver = tf.saved_model.builder.SavedModelBuilder(export_dir)

model_name = 'mnist_tf_basic'
model_name_save_teacher = 'mnist_teacher'
model_name_save_student_trained = 'mnist_student'

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


def Create_Data(session):
  batch_data = []
  batch_labels = []
  count_in_batch = 0
  minibatch_num = 0
  disc_data = []
  disc_label = []
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
      
      logit_t, logit_s = session.run([logits_teacher_eval, logits_student], feed_dict=feed_dict)

      for indx in range(logit_t.shape[0]):
        disc_data.append(logit_t[indx])
        disc_label.append([1,0])
        disc_data.append(logit_s[indx])
        disc_label.append([0,1])      


  images.close()
  labels.close()

  temp = list(zip(disc_data, disc_label))
  random.shuffle(temp)

  disc_data, disc_label = zip(*temp)

  disc_data = np.array(disc_data)
  disc_label = np.array(disc_label)

  disc_train_data = disc_data[:train_disc_size]
  disc_train_label = disc_label[:train_disc_size]

  disc_test_data = disc_data[train_disc_size:disc_data.shape[0]]
  disc_test_label = disc_label[train_disc_size:disc_data.shape[0]]

  print (disc_data.shape)
  print (disc_label.shape)
  print (disc_train_data.shape)
  print (disc_train_label.shape)
  print (disc_test_data.shape)
  print (disc_test_label.shape)

  disc_train_data = np.concatenate((disc_train_data, disc_train_label),axis=1)
  disc_test_data = np.concatenate((disc_test_data, disc_test_label),axis=1)
  
  # print (disc_train_data.shape)
  # print (disc_train_data[0])
  np.savetxt(fname=dir_disc_data + 'disc_train.txt',X=disc_train_data,delimiter=',')
  np.savetxt(fname=dir_disc_data + 'disc_test.txt',X=disc_test_data,delimiter=',')

  # np.savetxt(fname=dir_disc_data + 'disc_train_y.txt',X=disc_train_label,delimiter=',')
  # np.savetxt(fname=dir_disc_data + 'disc_test.txt',X=disc_test_data,delimiter=',')
  # np.savetxt(fname=dir_disc_data + 'disc_test_y.txt',X=disc_test_label,delimiter=',')
        

batch_size = 500
patch_size_teacher = 5
patch_size_student = 5
depth_teacher = 64
depth_student = 16
num_hidden_teacher = 1000
num_hidden_student = 200

train_disc_size = 90000

alpha = 10
beta = 0.001

# def make_student_graph_KD():
graph_create_data = tf.Graph()

with graph_create_data.as_default():
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
    
  logits_teacher_eval = teacher_model_eval(tf_train_dataset)
  logits_student = student_model(tf_train_dataset)

  prediction_teacher_eval = tf.nn.softmax(logits_teacher_eval)
  prediction_student = tf.nn.softmax(logits_student)

  


def create_disc_data():
  with tf.device(current_device):
    # graph_discriminator = make_student_graph_KD()

    with tf.Session(graph=graph_create_data) as session:
      tf.global_variables_initializer().run()
      
      saver = tf.train.Saver(var_list=teacher_parameters)
      saver.restore(session, export_dir_teacher + model_name_save_teacher)

      saver = tf.train.Saver(var_list=student_parameters)
      saver.restore(session, export_dir_student + model_name_save_student_trained)

      print ("Testing Teacher for sanity check")
      acc, w = test_accuracy(session)
      print('Teacher : Number of wrong classificiation: %d Test accuracy: %.1f%%' % (w, acc))

      print ("Testing Student for sanity check")
      acc, w = test_accuracy(session, False)
      print('Student : Number of wrong classificiation: %d Test accuracy: %.1f%%' % (w, acc))

      Create_Data(session)
      

create_disc_data()




  
    


  
