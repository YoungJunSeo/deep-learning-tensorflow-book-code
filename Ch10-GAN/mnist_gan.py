# -*- coding: utf-8 -*-
# GAN(Generative Adversarial Networks)을 이용한 MNIST 데이터 생성
# Reference : https://github.com/TengdaHan/GAN-TensorFlow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# MNIST 데이터를 불러옵니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 8x8개의 생성된 MNIST 이미지를 보여주는 plot 함수를 정의합니다.
def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28))
    return fig

# 설정값들을 선언합니다.
num_epoch = 100000
batch_size = 64
num_input = 28 * 28
num_latent_variable = 100
num_hidden = 128
learning_rate = 0.001

# 플레이스 홀더를 선언합니다.
X = tf.placeholder(tf.float32, [None, num_input])               # 인풋 이미지
z = tf.placeholder(tf.float32, [None, num_latent_variable])     # Latent Variable

# Generator를 생성하는 함수를 정의합니다.
# Inputs:
#   X : 인풋 이미지
#   num_hidden : 히든 레이어의 노드 개수
# Output:
#   generated_mnist_image : 생성된 MNIST 이미지
def build_generator(X, num_hidden):
    with tf.variable_scope('generator'):
        # 히든 레이어 설정
        W1 = tf.get_variable('G_W1', shape=[num_latent_variable, num_hidden], initializer=tf.random_normal_initializer(stddev=5e-2))
        b1 = tf.get_variable('G_b1', shape=[num_hidden], initializer=tf.constant_initializer())
        fc_1 = tf.nn.relu((tf.matmul(X, W1) + b1))
        # 아웃풋 레이어 설정
        W2 = tf.get_variable('G_W2', shape=[num_hidden, num_input], initializer=tf.random_normal_initializer(stddev=5e-2))
        b2 = tf.get_variable('G_b2', shape=[num_input], initializer=tf.constant_initializer())
        fc_2 = tf.matmul(fc_1, W2) + b2
        generated_mnist_image = tf.nn.sigmoid(fc_2)

        return generated_mnist_image

# Discirimantor를 생성하는 함수를 정의합니다.
# Inputs:
#   X : 인풋 이미지
#   num_hidden : 히든 레이어의 노드 개수
# Output:
#   predicted_value : Discriminator가 판단한 True(1) or Fake(0)
#   logits : sigmoid를 씌우기전의 출력값
def build_discriminator(X, num_hidden, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # 히든 레이어 설정
        W1 = tf.get_variable('D_W1', shape=[num_input, num_hidden], initializer=tf.random_normal_initializer(stddev=5e-2))
        b1 = tf.get_variable('D_b1', shape=[num_hidden], initializer=tf.constant_initializer())
        fc_1 = tf.nn.relu((tf.matmul(X, W1) + b1))
        # 아웃풋 레이어 설정
        W2 = tf.get_variable('D_W2', shape=[num_hidden, 1], initializer=tf.random_normal_initializer(stddev=5e-2))
        b2 = tf.get_variable('D_b2', shape=[1], initializer=tf.constant_initializer())
        logits = tf.matmul(fc_1, W2) + b2
        predicted_value = tf.nn.sigmoid(logits)

        return predicted_value, logits

# 생성자(Generator)를 선언합니다.
G = build_generator(z, num_hidden)

# 구분자(Discriminator)를 선언합니다.
D_real, D_real_logits = build_discriminator(X, num_hidden, reuse=False)
D_fake, D_fake_logits = build_discriminator(G, num_hidden, reuse=True)

# Discriminator의 손실 함수를 정의한다.
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
d_loss = d_loss_real + d_loss_fake

# Generator의 손실 함수를 정의합니다.
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                        (logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

tvar = tf.trainable_variables()
dvar = [var for var in tvar if 'discriminator' in var.name]
gvar = [var for var in tvar if 'generator' in var.name]

# Discriminator와 Generator의 Optimizer를 정의합니다.
d_train_step = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=dvar)
g_train_step = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=gvar)

# 생성된 이미지들을 저장할 generated_outputs 폴더를 생성합니다.
num_img = 0
if not os.path.exists('generated_output/'):
    os.makedirs('generated_output/')

with tf.Session() as sess:
    # 변수들에 초기값을 할당합니다.
    sess.run(tf.global_variables_initializer())

    # num_epoch 횟수만큼 최적화를 수행합니다.
    for i in range(num_epoch):
        # MNIST 이미지를 batch_size만큼 불러옵니다. 
        batch_X, _ = mnist.train.next_batch(batch_size)
        # Latent Variable의 인풋으로 사용할 noise를 Uniform Distribution에서 batch_size만큼 샘플링합니다.
        batch_noise = np.random.uniform(-1., 1., [batch_size, 100])

        # 500번 반복할때마다 생성된 이미지를 저장합니다.
        if i % 500 == 0:
            samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64, 100])})
            fig = plot(samples)
            plt.savefig('generated_output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)

        # Discriminator 최적화를 수행하고 Discriminator의 손실함수를 return합니다.
        _, d_loss_print = sess.run([d_train_step, d_loss],
                                   feed_dict={X: batch_X, z: batch_noise})

        # Generator 최적화를 수행하고 Generator 손실함수를 return합니다.
        _, g_loss_print = sess.run([g_train_step, g_loss],
                                   feed_dict={z: batch_noise})

        # 100번 반복할때마다 Discriminator의 손실함수와 Generator 손실함수를 출력합니다.
        if i % 100 == 0:
            print('반복(Epoch): %d, Generator 손실함수(g_loss): %f, Discriminator 손실함수(d_loss): %f' % (i, g_loss_print, d_loss_print))