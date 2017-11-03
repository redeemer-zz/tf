
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import numpy as np
from tf_honu import *
from tqdm import tqdm
import os
from torchcrayon import *
from time import sleep



time = np.arange(0,100,0.1, dtype=np.float32)
data = np.sin(time)
data += np.random.uniform(-0.03, 0.03, len(time))
# data[540] += 0.3
# y = data
# y = np.arange(10)

# N = 800
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
pred = 3

data, info = normalize(data)

train_n = 400
test_n = 400
honu_order = 1

x_train = swv(data, wlen)[:train_n, :]
y_train = swo(data, wlen, pred)[:train_n, :]
x_train, _ = ioa(x_train, y_train)


x_train = make_honu_batch(x_train, honu_order, True)


x = Variable(torch.Tensor(x_train))
y = Variable(torch.Tensor(y_train))

x_test = swv(data, wlen)[test_n:, :]
y_test = swo(data, wlen, pred)[test_n:, :]
x_test, _ = ioa(x_test, y_test)


x_test = make_honu_batch(x_test, honu_order, True)

x_t = Variable(torch.Tensor(x_test))
y_t = Variable(torch.Tensor(y_test))


class Model(torch.nn.Module):
    
    def __init__(self, w_num):
        super(Model, self).__init__()
        
        self.weights = Parameter(torch.unsqueeze(torch.Tensor(xavier_init(w_num)), 1), 
                                 requires_grad=True)
    
    def forward(self, x):
        net_input = x.mm(self.weights)
        return net_input
        
model = Model(len(x[0]))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

max_loss = 10e-13

n_i = 10000



def training(model, optimizer, x, y, max_loss, max_iter, loss_div = 10e6):
    optimizer.zero_grad()
    outputs = model(x)
        
    loss = criterion(outputs, y)

    loss_v = []
    weights = np.zeros((n_i, len(x.data[0])))
    for i in tqdm(range(n_i)):
        optimizer.zero_grad()
        outputs = model(x)
        
        loss = criterion(outputs, y)
        loss_v.append(loss.data[0])
        weights[i, :]  = (model.weights.data.numpy()[:,0])

        if loss.data.numpy() < max_loss:
            break
        
        if np.isnan(loss.data.numpy()) or np.isinf(loss.data.numpy()):
            print('Nan in {0}'.format(i))
            raise Exception('Divergation')
        
        if len(loss_v) > 10:
            if np.mean(loss_v[-10:-1]) > loss_div:
                print('\r\r')
                raise Exception('Divergation 2')  
       

        loss.backward()        
        optimizer.step()
    return loss_v, weights


l_r = 0.007
min_lr = 0.0001
lr_step = 0.0001


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
            outputs = model(training_set_in)
            if training_set_xaxis is not None:
                plt.plot(training_set_xaxis, training_set_out, 'r', label = 'HONU output')
            else:
                plt.plot(training_set_out, 'r', label = 'HONU output')
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
    model_output = model(x)
    try:
        loss_function = {'E': lambda x, y: (x.data.numpy()-y)
                        }[lossfcn]
    except KeyError as e:
        raise Exception('Unknown less evaluation method {0}', loss)
    loss = loss_function(model_output, y)

    fig = plt.figure()
    plt.subplot('211')
    plt.plot(y, label = 'Test set output')
    plt.plot(y, 'r', label = 'Model output')
    plt.legend()
    plt.subplot('212')
    plt.plot(loss, label = 'Loss')
    plt.legend()

    return fig



while True:
    del(model)
    del(optimizer)
    model = Model(len(x[0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=l_r)
    try:
        loss_v, weights = training(model, optimizer, x, y, max_loss, n_i)
        break
    except Exception as inst:
        print(inst)
        # os.system('clear')   
        l_r -= lr_step
        if l_r < min_lr:
            raise Exception('Divergation for all learning rates.')
        print('-------------------------------------------\n')
        print('Starting with learning rate: {0}\n'.format(l_r))
        continue



# CRAYON - slow!

# expt = get_experiment(name='s2s,opt=SGD')

# for i in tqdm(range(len(loss2))):
#     expt.add_scalar_dict({'loss_mxe/train': loss2[i]})
#     sleep(0.001)



outputs_t = model(x_t)
loss_t = criterion(outputs_t, y_t)

# train = plt.figure()
# plt.plot(y_train)


# train_plt_step = 20
# for idx, line in enumerate(train_run):
#     if idx % train_plt_step == 0:
#         plt.plot(line, 'r', alpha = idx/(len(train_run)))

# print(model.weights.data)
print('Error on training data: ', loss_v[-1])
print('Error on test data: ', loss_t.data.numpy())
# plt.figure()
# plt.plot(loss2)
# plt.figure()
# plt.plot(y_test)
# plt.plot(outputs_t.data.numpy(), 'r')
# print(type(x))
plot_training(model, loss_v, ind = x, outd = y.data.numpy(), w = np.array(weights))
plot_test(model, x_t, y_t.data.numpy())

plt.show()