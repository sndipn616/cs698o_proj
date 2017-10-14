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
export_dir = 'MNIST_Model_Teacher/'
temp_dir = 'MNIST_Models/'
# model_saver = tf.saved_model.builder.SavedModelBuilder(export_dir)

model_name_save_teacher = 'mnist_paper_teacher'
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
      [predictions] = session.run([prediction_eval], feed_dict=feed_dict)

      c, t, w = num_correct_total(predictions, new_batch_labels)
      correct += c
      total += t
      wrong += w

  testImages.close()
  testLabels.close()

  return 100.0 * correct / total, wrong

def Train_Teacher(session):  
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
        _, l, predictions = session.run([optimizer, loss, prediction_train], feed_dict=feed_dict)

        if minibatch_num % 100 == 0:
          print('Minibatch loss at step %d: %f' % (minibatch_num, l))          
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, new_batch_labels))

    images.close()
    labels.close()
        

  model_saver = tf.train.Saver()
  model_saver.save(session, export_dir + model_name_save_teacher, write_meta_graph=True)

  acc, w = test_accuracy(session)
  print('Number of wrong classificiation: %d Test accuracy: %.1f%%' % (w, acc))



batch_size = 100
patch_size = 3
prob = 0.5
num_hidden = 1200
num_epochs = 3
beta = 0.001


graph_teacher = tf.Graph()

with graph_teacher.as_default():

    '''Input data'''
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size * num_channels), name='x')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='y')
    # tf_valid_dataset = tf.constant(valid_dataset)
    # tf_test_dataset = tf.constant(test_dataset)
    # tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    # tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    '''Variables For Teacher'''
    # Input to Hidden1 Layer    
    layer1_weights_teacher = tf.Variable(tf.truncated_normal([image_size * image_size * num_channels, num_hidden], stddev=0.1), name='l1wt')
    layer1_biases_teacher = tf.Variable(tf.zeros([num_hidden]), name='l1bt')
    
    # Hidden1 to Hidden2 Layer
    layer2_weights_teacher = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1), name='l2wt')
    layer2_biases_teacher = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='l2bt')

    # Hidden2 to Output Layer
    layer3_weights_teacher = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='l3wt')
    layer3_biases_teacher = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='l3bt')

    

    # data = tf_train_dataset
    '''Teacher Model for Training'''
    def teacher_model_train(data):
      out = tf.matmul(tf_train_dataset, layer1_weights_teacher) + layer1_biases_teacher
      out = tf.nn.dropout(tf.nn.relu(out), prob)

      out = tf.matmul(out, layer2_weights_teacher) + layer2_biases_teacher
      out = tf.nn.dropout(tf.nn.relu(out), prob)

      out = tf.matmul(out, layer3_weights_teacher) + layer3_biases_teacher

      return out

    def teacher_model_eval(data):
      out = tf.matmul(tf_train_dataset, layer1_weights_teacher) + layer1_biases_teacher
      out = tf.nn.relu(out)

      out = tf.matmul(out, layer2_weights_teacher) + layer2_biases_teacher
      out = tf.nn.relu(out)

      out = tf.matmul(out, layer3_weights_teacher) + layer3_biases_teacher

      return out
       
    # logits = tf.matmul(hidden, layersm_weights_teacher) + layersm_biases_teacher
    '''Training computation'''   
    logits_train = teacher_model_train(tf_train_dataset)
    logits_eval = teacher_model_eval(tf_train_dataset)

    tf.add_to_collection("teacher_model_logits", logits_eval)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_train)) \
    + beta*tf.nn.l2_loss(layer1_weights_teacher) + beta*tf.nn.l2_loss(layer2_weights_teacher) + beta*tf.nn.l2_loss(layer3_weights_teacher)

    '''Optimizer'''
    # Learning rate of 0.05
    # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss) 

    '''Predictions for the training, validation, and test data'''
    prediction_train = tf.nn.softmax(logits_train)
    prediction_eval = tf.nn.softmax(logits_eval)

    tf.add_to_collection("teacher_model_prediction", prediction_eval)



with tf.device(current_device):  
  with tf.Session(graph=graph_teacher) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    if os.path.isfile(export_dir + model_name_save_teacher + '.meta'):
      saver = tf.train.Saver()
      saver.restore(session, export_dir + model_name_save_teacher)

    Train_Teacher(session)  

  
    


  
