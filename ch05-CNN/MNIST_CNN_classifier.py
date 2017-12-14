# -*- coding: utf-8 -*-

# Convolutional Neural Networks(CNN)을 이용한 MNIST 분류기(Classifier)

# 필요한 라이브러리들을 임포트
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def build_CNN_classifier(x):
  """ 
    CNN Model을 생성한다.
  """

  # MNIST 데이터를 3차원 형태로 reshape한다. MNIST 데이터는 grayscale 이미지라서 3번째차원(컬러채널)의 값은 1이다.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # 첫번째 Convolution Layer 
  # 5x5 Kernel Size를 가진 32개의 Filter를 적용한다.
  # 28x28x1 -> 28x28x32
  W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # 첫번째 Pooling Layer - Max Pooling을 이용해서 이미지의 크기를 1/2로 downsample한다.
  # 28x28x32 -> 14x14x32
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 두번째 Convolutional Layer 
  # 5x5 Kernel Size를 가진 64개의 Filter를 적용한다.
  # 14x14x32 -> 14x14x64
  W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # 두번째 Pooling Layer - Max Pooling을 이용해서 이미지의 크기를 1/2로 downsample한다.
  # 14x14x64 -> 7x7x64
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Fully Connected Layer
  # 7x7 크기를 가진 64개의 activation map을 1024개의 특징들로 변환한다.
  # 7x7x64 -> 1024
  W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Output Layer
  # 1024개의 특징들(feature)을 10개의 클래스-숫자 0-9-로 변환한다.
  # 1024 -> 10
  W_output = tf.Variable(tf.truncated_normal([1024, 10], stddev=5e-2))
  b_output = tf.Variable(tf.constant(0.1, shape=[10]))

  h_output = tf.matmul(h_fc1, W_output) + b_output
  y_pred = tf.nn.softmax(h_output)
  return y_pred

#MNIST 데이터를 불러온다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 인풋, 아웃풋 데이터의 크기를 설정한다.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Convolutional Neural Networks(CNN) 그래프를 생성한다.
y_pred = build_CNN_classifier(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, AdamOptimizer를 이용해서 비용 함수를 최소화한다.
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)

# 정확도를 측정한다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션을 열어 실제 학습을 진행한다.
with tf.Session() as sess:
  # 모든 변수들을 초기화한다. 
  sess.run(tf.global_variables_initializer())

  # 20000 Step만큼 학습을 진행한다.
  for i in range(20000):
    # 50개씩 MNIST 데이터를 불러온다.
    batch = mnist.train.next_batch(50)
    # 100 Step마다 training 데이터셋에 대한 정확도를 출력한다.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
      print('step %d, 트레이닝 데이터 정확도 : %g' % (i, train_accuracy))
    # 학습을 진행한다.
    train_step.run(feed_dict={x: batch[0], y: batch[1]})

  # 학습이 끝나면 테스트 데이터에 대한 정확도를 출력한다.
  print('테스트 데이터 정확도 : %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))




