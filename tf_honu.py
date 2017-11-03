import numpy as np
import matplotlib.pyplot as plt
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

def plot_training(model, loss, *args, **kwargs):
    training_set_in = None
    training_set_out = None
    training_set_xaxis = None
    weights = None
    m = None
    if kwargs is not None:
        for key, value in kwargs.items():
            if key in ('training_set_input', 'ind'):
                training_set_in = value
            if key in ('training_set_output', 'outd'):
                training_set_out = value
            if key in ('training_set_xaxis', 'datax'):
                training_set_xaxis = value
            if key in ('weights', 'w'):
                weights = value
            if key in ('message', 'm'):
                message = value
    sub_plot_n = 1
    sub_plot = 1
    if (training_set_in is not None) or (training_set_out is not None):
        sub_plot_n += 1
    if weights is not None:
        sub_plot_n += 1

    fig = plt.figure()
    if (training_set_in is not None) or (training_set_out is not None):

        plt.subplot('{0}1{1}'.format(sub_plot_n, sub_plot))
        if training_set_out is not None:
            if training_set_xaxis is not None:
                plt.plot(training_set_xaxis, training_set_out, 'b', label = 'Training set output')
            else:
                plt.plot(training_set_out, 'b', label = 'Training set output')
        if training_set_in is not None:
            outputs = model(training_set_in).data.tolist()
            if training_set_xaxis is not None:
                plt.plot(training_set_xaxis, outputs, 'r', label = 'HONU output')
            else:
                plt.plot(outputs, 'r', label = 'HONU output')
        sub_plot += 1
        plt.legend()


    plt.subplot('{0}1{1}'.format(sub_plot_n, sub_plot))
    plt.plot(loss, label = 'Loss')
    plt.legend()
    sub_plot += 1
    if weights is not None:
        plt.subplot('{0}1{1}'.format(sub_plot_n, sub_plot))

        colormap = plt.cm.gist_ncar
        plt.plot(weights)
        legend = []
        for idx in range(len(weights)):
            legend.append('w{0}'.format(idx))
        plt.legend(legend, ncol=7, loc='upper center')
    plt.tight_layout()
    return fig

def plot_test(model, x, y, lossfcn='E'):
    model_output = model(x).data.tolist()
    try:
        loss_function = {'E': lambda x, y: (x-y)
                        }[lossfcn]
    except KeyError as e:
        raise Exception('Unknown less evaluation method {0}', loss)
    loss = loss_function(model_output, y)

    fig = plt.figure()
    plt.subplot('211')
    plt.plot(y, label = 'Test set output')
    plt.plot(model_output, 'r', label = 'Model output')
    plt.legend()
    plt.subplot('212')
    plt.plot(loss, label = 'Loss')
    plt.legend()

    return fig
