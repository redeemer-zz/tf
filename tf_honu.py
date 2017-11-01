import numpy as np
from copy import copy

def add_bias(X):
    if len(X.shape) == 1:
        return np.concatenate((np.array([1]), X))
    else:
        bias_vec = np.ones([X.shape[0],1])
        return np.hstack((bias_vec, X))

def honu_xcol_iteration(input_vec, output, order, start = 0, submul = []):
    if type(input_vec) == list:
        input_vec = np.array(input_vec)
    order_it = order - 1
    for idx, val in enumerate(input_vec[start:], start):
        submul_it = copy(submul)
        submul_it.append(val)
        if order_it > 0:
            honu_xcol_iteration(input_vec, output, order_it, idx, submul_it)
        else:
            output.append(np.prod(submul_it))

def make_honu_row(input_row, order, bias = True):
    ivb = add_bias(input_row)
    honu = []
    honu_xcol_iteration(ivb, honu, order)
    if bias:
        return honu
    else:
        return honu[1:]

def make_honu_batch(input_matrix, order, bias = True):
    if type(input_matrix) in (list, tuple):
        input_matrix = np.array(input_matrix)

    honu_elements = 0
    n = input_matrix.shape[1]
    for j in range(order):
        honu_elements += int(np.math.factorial(n+j)/(np.math.factorial(j+1)*(np.math.factorial(n-1))))

    if bias: honu_elements += 1

    output = np.zeros((input_matrix.shape[0], honu_elements))
    
    for idx, row in enumerate(input_matrix):
        output[idx, :] = make_honu_row(row, order, bias)
    
    return output

def honu_net(HONUs = []):
    net = HONUs[0]
    for idx in range(1, len(HONUs)):
        net = np.hstack((net, HONUs[idx]))
    return net

def xavier_init(n):
    x = np.sqrt(6./(n))
    return np.random.uniform(-x, x, n).astype(np.float32)

def normalize(X, axis = 0):
    if len(X.shape) > 1:
        means = np.mean(X, axis)
        stds = np.std(X, axis)
        norm_equation = lambda row: (row-means) / stds
        return (np.apply_along_axis(norm_equation, axis, X)), [means, stds]
    else:
        mean = np.mean(X)
        std = np.std(X)
        return (X-mean) / std, [mean, std]

def sliding_window_vector(X, history):
    return np.array([X[i:(i+history)] for i in range(len(X)) if len(X[i:(i+history)]) == history])

def swv(X, history): return sliding_window_vector(X, history)

def sliding_window_output(Y, history, prediction): return swv(Y[history+prediction-1:], 1)

def swo(Y, history, prediction): return sliding_window_output(Y, history, prediction)

def input_output_aligment(input, output): 
    return input[:output.shape[0]], output

def ioa(input, output): return input_output_aligment(input, output)
