import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from time import time
import __main__
from tensorflow.python.client import timeline


### TENSORBOARD setup
tensor_board_ouput_dir = '.\\tensorboard\\summaries\\%s\\%s\\' % (__main__.__file__, time())

### Save tensor as separated scalars to summary
def tensor_1D_summary(tensor, name='defaultTensorName'):
    for i in range(tensor.get_shape()[0]):
        tf.summary.scalar(name + str(i), tensor[i])

def add_bias(X):
    bias_vec = np.ones([X.shape[0],1])
    return np.hstack((bias_vec, X))

def make_qnu_line(line):
    nXh = np.sum(range(len(line)+1)) 
    Xh = np.zeros(nXh)

    idxh = 0
    for idx, xi in enumerate(line):
        for xj in line[idx:]:
            Xh[idxh] = xi*xj
            idxh += 1 

    return Xh

def make_lnu(X):
    return add_bias(X)

def make_qnu(X, bias = False):
    nXh = np.sum(range(X.shape[1]+1)) 
    Xh = np.zeros([X.shape[0], nXh])

    for row_idx in range(X.shape[0]):
        Xh[row_idx,:] = make_qnu_line(X[row_idx,:])
    if bias: Xh = add_bias(Xh)
    return Xh

def honu_net(HONUs = []):
    net = HONUs[0]
    for idx in range(1, len(HONUs)):
        net = np.hstack((net, HONUs[idx]))
    return net


max_loss = 1e-12

### Training data
x_t = np.array([[1, 2, 3], [1, 3, 3], [1, 3, 4,], [1, 2, 3]])
target_weights = np.array([0.5, 1, 2])
y_train = x_t.dot(target_weights)

# print(honu_net([x_train, make_qnu(x_train)]))
x_train_lnu = make_lnu(x_t)
x_train = make_qnu(x_t, True)
# x_train = honu_net([x_train_lnu, x_train_qnu])

### Placeholders
X = tf.placeholder(tf.float32,[x_train.shape[0], x_train.shape[1]], name='X')
y = tf.placeholder(tf.float32, name='y')

### Weights
with tf.name_scope('weights'):
    W = tf.get_variable("W", shape=[x_train.shape[1]],
                    initializer=xavier_initializer())
    tensor_1D_summary(W, 'w')

### HONU equation
with tf.name_scope('model'):
    linear_model = tf.reshape(tf.matmul(X,tf.expand_dims(W,1)), [-1])

### Loss
with tf.name_scope('loss'):
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
tf.summary.scalar('loss', loss)

### Training algorithm
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(0.1).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

### Merge all summaries
merged = tf.summary.merge_all()

with tf.Session() as sess:
    
    init = tf.global_variables_initializer().run()
    
    summary_writer = tf.summary.FileWriter(tensor_board_ouput_dir, sess.graph)
    summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START))
    
    for i in range(500):
    
        # if i % 100 == 99:  # Record execution stats and iteration 
        #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #     run_metadata = tf.RunMetadata()
        #     summary, _ = sess.run([merged, train],
        #                           {X: x_train, y: y_train},
        #                           options=run_options,
        #                           run_metadata=run_metadata)
        #     summary_writer.add_run_metadata(run_metadata, 'step_%d' % i)
        #     summary_writer.add_summary(summary, i)
    
        # else:   # Record iteration
            summary, _ = sess.run([merged, train], {X: x_train, y: y_train})
            error = sess.run(loss, {X: x_train, y: y_train})
            weight = sess.run(W)
            summary_writer.add_summary(summary, i)
            # print('Iteration {0}, error: {1}'.format(i, error))

            if error < max_loss: break

print('Iteration {0}, error: {1}'.format(i, error))
print(weight)
summary_writer.close()
