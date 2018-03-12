# -*- coding: utf-8 -*-
# Embedding & tf.nn.embedding_lookup 예제

import tensorflow as tf
import numpy as np

vocab_size = 100
embedding_size = 25

# 인풋데이터를 받기 위한 플레이스홀더를 선언합니다.
inputs = tf.placeholder(tf.int32, shape=[None])
 
# 인풋데이터를 변환하기 위한 Embedding Matrix를 선언합니다.  
embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size]), dtype=tf.float32)
# Embedding 된 inputs 데이터를 리턴합니다.
# tf.nn.embedding_lookup :
# int32나 int64 형태의 스칼라 형태의 인풋데이터를 vocab 사이즈만큼의 ebmedding된 vector로 변환 
embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)

# 세션을 열고 모든 변수에 초기값을 할당합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tf.nn.embedding_lookup 테스트 케이스 1
input_data = np.array([7])
print("Embedding 전 인풋데이터 : ")
# [1]
print(input_data)
print(input_data.shape)
print("Embedding 결과 : ")
# [1, 25]
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs : input_data})[0].shape)


# tf.nn.embedding_lookup 테스트 케이스 2
input_data = np.array([7, 11, 67, 42, 21])
print("Embedding 전 인풋데이터 : ")
# [5]
print(input_data)
print(input_data.shape)
print("Embedding 결과 : ")
# [5, 25]
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs : input_data})[0].shape)

