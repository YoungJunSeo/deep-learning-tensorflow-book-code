# -*- coding: utf-8 -*-

"""
CIFAR-10 Convolutional Neural Networks(CNN) Example

next_batch function is copied from edo's answer

https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data

Author : solaris33
Project URL : http://solarisailab.com/archives/2325
"""

import tensorflow as tf
import numpy as np

from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data


def build_CNN(x):
  """CIFAR-10 이미지를 분류하기 위한 Convolutional Neural Networks 그래프를 생성한다.
  인자들(Args):
    x: (N_examples, 32, 32, 3) 차원을 가진 input tensor, CIFAR-10 데이터는 32x32 크기의 컬러이미지이다.
  리턴값들(Returns):
    tuple (y, keep_prob). y는 (N_examples, 10)형태의 숫자(0-9) tensor이다. 
    keep_prob는 dropout을 위한 scalar placeholder이다.
  """
  # 입력 이미지
  x_image = x

  # 첫번째 convolutional layer - 하나의 grayscale 이미지를 32개의 특징들(feature)으로 맵핑(maping)한다.
  W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # Pooling layer - 2X만큼 downsample한다.
  #h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 두번째 convolutional layer -- 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)한다.
  W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # 두번째 pooling layer.
  #h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 세번째 convolutional layer
  W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=5e-2))
  b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

  # 네번째 convolutional layer
  W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=5e-2))
  b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
  h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

  # 다섯번째 convolutional layer
  W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=5e-2))
  b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

  # Fully Connected Layer 1 -- 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 된다.
  # 이를 384개의 특징들로 맵핑(maping)한다.
  W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 128, 384], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

  h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

  # Dropout - 모델의 복잡도를 컨트롤한다. 특징들의 co-adaptation을 방지한다.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

  # 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)한다.
  W_fc2 = tf.Variable(tf.truncated_normal([384, 10], stddev=5e-2))
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
  y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

  #y_pred = tf.nn.softmax(y_conv)
  y_pred = y_conv
  return y_pred, keep_prob


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# CIFAR-10 데이터를 불러온다. 
(x_train, y_train), (x_test, y_test) = load_data()

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환한다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

# Input과 Ouput의 차원을 가이드한다.
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10])

# Convolutional Neural Networks(CNN) 그래프를 생성한다.
y_conv, keep_prob = build_CNN(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer 이용해서 비용 함수를 최소화한다.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)

# 정확도를 측정한다.
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 하이퍼 파라미터를 정의한다.
max_steps = 10000 # 최대 몇 step을 학습할지를 정한다. 


with tf.Session() as sess:
  # 모든 변수들을 초기화한다. 
  sess.run(tf.global_variables_initializer())
  
  # 20000번 학습(training)을 진행한다.
  for i in range(max_steps):
    batch = next_batch(128, x_train, y_train_one_hot.eval())

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력한다.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={ x: batch[0], y_: batch[1], keep_prob: 1.0})
      loss = cross_entropy.eval(feed_dict={ x: batch[0], y_: batch[1], keep_prob: 1.0})

      print('step %d, training accuracy %g, loss %g' % (i, train_accuracy, loss))
    # 20% 확률의 Dropout을 이용해서 학습을 진행한다.
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

  test_batch = next_batch(10000, x_test, y_test_one_hot.eval())
  # 테스트 데이터에 대한 정확도를 출력한다.
  print('test accuracy %g' % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))




