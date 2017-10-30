import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from time import time
import __main__
from tensorflow.python.client import timeline


### TENSORBOARD setup
tensor_board_ouput_dir = './tensorboard/summaries/%s/%s/' % (__main__.__file__, time())

### Save tensor as separated scalars to summary
def tensor_1D_summary(tensor, name='defaultTensorName'):
    for i in range(tensor.get_shape()[0]):
        tf.summary.scalar(name + str(i), tensor[i])


### Training data
x_train = np.array([[1, 2, 3], [1,3,3], [1, 3, 4,], [1, 2, 3]])
target_weights = np.array([0.5, 1, 2])
y_train = x_train.dot(target_weights)




### Placeholders
X = tf.placeholder(tf.float32,[4,3], name='X')
y = tf.placeholder(tf.float32, name='y')

### Weights
with tf.name_scope('weights'):
    W = tf.get_variable("W", shape=[3],
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
    # train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

### Merge all summaries
merged = tf.summary.merge_all()

with tf.Session() as sess:
    
    init = tf.global_variables_initializer().run()
    
    summary_writer = tf.summary.FileWriter(tensor_board_ouput_dir, sess.graph)
    summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START))
    
    for i in range(1000):
    
        if i % 100 == 99:  # Record execution stats and iteration 
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train],
                                  {X: x_train, y: y_train},
                                  options=run_options,
                                  run_metadata=run_metadata)
            summary_writer.add_run_metadata(run_metadata, 'step_%d' % i)
            summary_writer.add_summary(summary, i)
    
        else:   # Record iteration
            summary, _ = sess.run([merged, train], {X: x_train, y: y_train})
            summary_writer.add_summary(summary, i)

summary_writer.close()
