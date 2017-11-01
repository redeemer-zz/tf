import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tf_honu import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import copy
import sympy as sp

max_loss = 1e-10





time = np.arange(0,30,0.1, dtype=np.float32)
data = np.sin(time)
y = data #+ np.random.uniform(-0.05, 0.05, len(time))
# y = data
# y = np.arange(10)

N = 300
a = 0.2
b = 0.8
c = 0.9
d = 20
h = 10.0
y = np.zeros(N)
y[0] = 0.1
for k in range(0,N-1):
    y[k+1]  = (c*y[k]) + ( (a*y[k-d]) / (b + ( y[k-d]**h)) )    

wlen = 11
pred = 3

y, _ = normalize(y)

train_n = 600

x_train = swv(y, wlen)[:train_n,:]
y_train = swo(y, wlen, 1)[:train_n,:]
x_train, _ = ioa(x_train, y_train)

x_train = make_honu_batch(x_train, 1, True)

# print(x_train)
# print(y_train)
# exit()
#plt.plot(time, y)
#plt.show()

### Training data
#x_t = np.array([[1, 2, 3], [2, 3, 3], [1, 3, 4,], [2, 2, 3]])
# x_t, _ = normalize(x_t)
#target_weights = np.array([0.5, 1, 2])
#y_train = x_t.dot(target_weights)+2

# print(honu_net([x_train, make_qnu(x_train)]))
# x_train = make_lnu(x_train)
# x_train = make_qnu(x_train, False)
# x_train = add_bias(x_train)
# print(x_train)

# x_train = honu_net([x_train_lnu, x_train_qnu])

X = tf.placeholder(tf.float32,[x_train.shape[0], x_train.shape[1]], name='X')
y = tf.placeholder(tf.float32, name='y')

# W = tf.get_variable("W", shape=[x_train.shape[1]],
#                     initializer=xavier_initializer())

W = tf.Variable(np.random.uniform(-1,1,x_train.shape[1]).astype(np.float32)) #tf.Variable(xavier_init(x_train.shape[1]))
print(W)
honu = tf.reshape(tf.matmul(X,tf.expand_dims(W,1)), [-1])
loss = tf.reduce_sum(tf.square(honu - y)) # sum of the squares
# train = tf.train.AdamOptimizer(0.5).minimize(loss)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
print('Training...')

error_mean = None
error_mean_stop = 10e-2
with tf.Session() as sess:
    
    init = tf.global_variables_initializer().run()
    
    try:
        for i in range(1):
            sess.run(train, {X: x_train, y: y_train})
            error = sess.run(loss, {X: x_train, y: y_train})
            weight = sess.run(W)

            if error_mean == None:
                error_mean = error
            else:
                error_mean = (29*error_mean + error) / 30
                if np.abs(error - error_mean) < error_mean_stop: break
            if error < max_loss: break
            print('Iteration {0}, error: {1}'.format(i, error))
    
    except KeyboardInterrupt:
        pass

    print('Iteration {0}, error: {1}'.format(i, error))
    # print(weight)
    plt.plot(y_train)
    plt.plot(sess.run(honu, {X: x_train, y: y_train}), 'r')
    plt.show()
