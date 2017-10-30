import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tf_honu import *

max_loss = 1e-10

def norm_equation_row(row, mean_vec, std_vec):
    return 

def normalize(X):
    means = np.mean(X, 0)
    stds = np.std(X, 0)
    norm_equation = lambda row: (row-means) / stds
    return (np.apply_along_axis(norm_equation, 1, X)), [means, stds]

### Training data
x_t = np.array([[1, 2, 3], [2, 3, 3], [1, 3, 4,], [2, 2, 3]])
# x_t, _ = normalize(x_t)
target_weights = np.array([0.5, 1, 2])
y_train = x_t.dot(target_weights)+2

# print(honu_net([x_train, make_qnu(x_train)]))
# x_train = make_lnu(x_t)
x_train = make_qnu(x_t, False)
x_train, _ = normalize(x_train)
x_train = add_bias(x_train)

# x_train = honu_net([x_train_lnu, x_train_qnu])

X = tf.placeholder(tf.float32,[x_train.shape[0], x_train.shape[1]], name='X')
y = tf.placeholder(tf.float32, name='y')

# W = tf.get_variable("W", shape=[x_train.shape[1]],
#                     initializer=xavier_initializer())
W = tf.Variable(xavier_init(x_train.shape[1]))
honu = tf.reshape(tf.matmul(X,tf.expand_dims(W,1)), [-1])
loss = tf.reduce_sum(tf.square(honu - y)) # sum of the squares
# train = tf.train.AdamOptimizer(0.5).minimize(loss)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
print('Training...')
with tf.Session() as sess:
    
    init = tf.global_variables_initializer().run()
    
    for i in range(5000):
        sess.run(train, {X: x_train, y: y_train})
        error = sess.run(loss, {X: x_train, y: y_train})
        weight = sess.run(W)

        if error < max_loss: break

print('Iteration {0}, error: {1}'.format(i, error))
print(weight)
