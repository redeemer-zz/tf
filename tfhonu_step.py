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





time = np.arange(0,100,0.1, dtype=np.float32)
data = np.sin(time)
# data += np.random.uniform(-0.01, 0.01, len(time))
data[390] += 0.3
# y = data
# y = np.arange(10)

# N = 600
# a = 0.2
# b = 0.8
# c = 0.9
# d = 20
# h = 10.0
# data = np.zeros(N)
# data[0] = 0.1
# for k in range(0,N-1):
#     data[k+1]  = (c*data[k]) + ( (a*data[k-d]) / (b + ( data[k-d]**h)) )    

wlen = 11
pred = 1

data, _ = normalize(data)

train_n = 400
test_n = 400
honu_order = 1

x_train = swv(data, wlen)[:train_n, :]
y_train = swo(data, wlen, pred)[:train_n, :]
x_train, _ = ioa(x_train, y_train)

x_train = make_honu_batch(x_train, honu_order, True)


x_test = swv(data, wlen)[test_n:, :]
y_test = swo(data, wlen, pred)[test_n:, :]
x_test, _ = ioa(x_test, y_test)

x_test = make_honu_batch(x_test, honu_order, True)


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

X_step = tf.placeholder(tf.float32,[x_train.shape[1]], name='X_step')
y_step = tf.placeholder(tf.float32, name='y_step')

X_test = tf.placeholder(tf.float32,[x_test.shape[0], x_test.shape[1]], name='X_test')
# W = tf.get_variable("W", shape=[x_train.shape[1]],
#                     initializer=xavier_initializer())
batch = tf.Variable(0)
W = tf.Variable(np.random.uniform(-1,1,x_train.shape[1]).astype(np.float32)) #tf.Variable(xavier_init(x_train.shape[1]))
print(W)
honu = tf.reshape(tf.matmul(X,tf.expand_dims(W,1)), [-1])
loss = tf.reduce_sum(tf.square(honu - y)) # sum of the squares

honu_step = tf.reshape(tf.matmul(tf.transpose(tf.expand_dims(X_step,1)),tf.expand_dims(W,1)), [-1])
loss_step = tf.reduce_sum(tf.square(honu_step - y_step)) # sum of the squares

honu_test = tf.reshape(tf.matmul(X_test, tf.expand_dims(W,1)), [-1])


# train = tf.train.AdamOptimizer(0.5).minimize(loss_step, global_step=batch)
train = tf.train.GradientDescentOptimizer(0.005).minimize(loss_step, global_step=batch)
print('Training...')

error_mean = None
error_mean_stop = 10e-2
plttrain = plt.figure()
plt.plot(y_train)
plttest = plt.figure()
plt.plot(y_test)


it_n = 11

with tf.Session() as sess:
    
    init = tf.global_variables_initializer().run()
    
    try:
        for i in range(it_n):
            weights = []
            for step, x_row in enumerate(x_train):
                sess.run(train, {X_step: x_row, y_step: y_train[step]})
                error = sess.run(loss_step, {X_step: x_row, y_step: y_train[step]})
                weight = sess.run(W)
                weights.append(weight)


                # print('Iteration {0}, step {2}, error: {1}'.format(i, error, step))
                
            error = sess.run(loss, {X: x_train, y: y_train})
            weight = sess.run(W)
            weights.append(weight)

            if error_mean == None:
                error_mean = error
            else:
                error_mean = (29*error_mean + error) / 30
                if np.abs(error - error_mean) < error_mean_stop: break
            if error < max_loss: break
            print('Iteration {0}, error: {1}'.format(i, error))
            
            if i%10 == 0:
                plt.figure(plttrain.number)
                plt.plot(sess.run(honu, {X: x_train, y: y_train}), 'r', alpha = 1/it_n*i)
        
    except KeyboardInterrupt:
        pass

    plt.figure(plttest.number)
    ytest = []
    for step, x_row in enumerate(x_test):
        if step < wlen:
            y_last = sess.run(honu_step, {X_step: x_row})
        else: 
            feed = np.concatenate((np.array([1]), np.array(ytest[-wlen:])))
            y_last = sess.run(honu_step, {X_step: feed})

        if step > 50:
            break
        ytest.append(y_last[0])
    plt.plot(sess.run(honu_test, {X_test: x_test, y: y_test}), 'r')
    plt.plot(ytest, 'g')
    plt.figure()
    plt.plot(np.array(weights))
    print('Iteration {0}, error: {1}'.format(i, error))
    # print(weight)
    plt.show()
