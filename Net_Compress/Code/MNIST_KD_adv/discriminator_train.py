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
export_dir = 'Disc_Model/'
model_name_save_disc = 'disc_model'
data_dir = 'Disc_data/'

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

image_size = 28
num_labels = 10
num_labels_disc = 2
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size * image_size * num_channels)).astype(np.float32)
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



def test_accuracy(session,teacher=True):  

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
      
      feed_dict = {tf_train_dataset : new_batch_data, tf_train_labels : new_batch_labels}
      if teacher:
        [predictions] = session.run([prediction_teacher_eval], feed_dict=feed_dict)
      else:
        [predictions] = session.run([prediction_student], feed_dict=feed_dict)

      c, t, w = num_correct_total(predictions, new_batch_labels)
      correct += c
      total += t
      wrong += w  

  return 100.0 * correct / total, wrong


def Train_Disc(session):
  for epoch in range(num_epochs_disc):
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

        if minibatch_num % 100 == 0:
          # print (type(l))
          print('Minibatch loss at step %d and epoch %d : %f' % (minibatch_num, epoch, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))

    images.close()
    labels.close()
        

  model_saver = tf.train.Saver(var_list=disc_parameters)
  model_saver.save(session, export_dir + model_name_save_disc, write_meta_graph=True)

  acc, w = test_accuracy(session, teacher=False)
  print('Student : alpha = %f, T = %d, Number of wrong classificiation: %d Test accuracy: %.1f%%' % (alpha, T, w, acc))

  

batch_size = 100
patch_size = 3
depth = 32
num_hidden_teacher = 1200
num_hidden_student = 800
num_hidden_disc = 100
num_labels_disc = 2
# num_epochs_teacher = 3
num_epochs_student = 5
T = 10
prob = 1

alpha = 10
beta = 0.001

# def make_student_graph_KD():
graph_discriminator = tf.Graph()

with graph_discriminator.as_default():

  '''Input data'''
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='x')
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels_disc), name='y')

  '''Variables For Discriminator'''
  # Input to Hidden1 Layer    
  layer1_weights_disc = tf.Variable(tf.truncated_normal([num_labels, num_hidden_disc], stddev=0.1), name='l1wd')
  layer1_biases_disc = tf.Variable(tf.zeros([num_hidden_disc]), name='l1bd')
  
  # Hidden1 to Hidden2 Layer
  layer2_weights_disc = tf.Variable(tf.truncated_normal([num_hidden_disc, num_hidden_disc], stddev=0.1), name='l2wd')
  layer2_biases_disc = tf.Variable(tf.constant(1.0, shape=[num_hidden_disc]), name='l2bd')

  # Hidden2 to Output Layer
  layer3_weights_disc = tf.Variable(tf.truncated_normal([num_hidden_disc, num_labels_disc], stddev=0.1), name='l3wd')
  layer3_biases_disc = tf.Variable(tf.constant(1.0, shape=[num_labels_disc]), name='l3bd')

  disc_parameters = [layer1_weights_disc, layer1_biases_disc, layer2_weights_disc, layer2_biases_disc, layer3_weights_disc, layer3_biases_disc]

  def discriminator_model(data):
    out = tf.matmul(tf_train_dataset, layer1_weights_disc) + layer1_biases_disc
    out = tf.nn.relu(out)

    out = tf.matmul(out, layer2_weights_disc) + layer2_biases_disc
    out = tf.nn.relu(out)

    out = tf.matmul(out, layer3_weights_disc) + layer3_biases_disc

    return out

     
  logits_disc = discriminator_model(tf_train_dataset)

  loss_disc = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_disc)) 

  '''Optimizer'''  
  optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_disc)
  
  '''Predictions for the training, validation, and test data'''
  prediction_disc = tf.nn.softmax(logits_disc)
  

  


def train_discriminator():
  with tf.device(current_device):
    # graph_discriminator = make_student_graph_KD()

    with tf.Session(graph=graph_discriminator) as session:
      tf.global_variables_initializer().run()
      
      # saver = tf.train.Saver(var_list=disc_parameters)
      # saver.restore(session, export_dir_teacher + model_name_save_teacher)

      






  
    


  
