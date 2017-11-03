
import matplotlib.pyplot as plt
from torch.autograd import Variable
# import torch.nn.functional as F
from torch.nn import Parameter
import torch
import numpy as np
from tf_honu import *
from tqdm import tqdm
import os
from time import sleep



# time = np.arange(0,100,0.1, dtype=np.float32)
# data = np.sin(time)
# data += np.random.uniform(-0.03, 0.03, len(time))
# data[540] += 0.3
# y = data
# y = np.arange(10)

N = 800
a = 0.2
b = 0.8
c = 0.9
d = 20
h = 10.0
data = np.zeros(N)
data[0] = 0.1
for k in range(0,N-1):
    data[k+1]  = (c*data[k]) + ( (a*data[k-d]) / (b + ( data[k-d]**h)) )    

data, info = normalize(data)

wlen = 11
pred = 1

train_n = 400
test_n = (400, -1)
honu_order = 1

x_train = swv(data, wlen)[:train_n, :]
y_train = swo(data, wlen, pred)[:train_n, :]
x_train, _ = ioa(x_train, y_train)
x_train = make_honu_batch(x_train, honu_order, True)
x = Variable(torch.Tensor(x_train))
y = Variable(torch.Tensor(y_train))

x_test = swv(data, wlen)[test_n[0]:test_n[1], :]
y_test = swo(data, wlen, pred)[test_n[0]:test_n[1], :]
x_test, _ = ioa(x_test, y_test)
x_test = make_honu_batch(x_test, honu_order, True)
x_t = Variable(torch.Tensor(x_test))
y_t = Variable(torch.Tensor(y_test))


class Model(torch.nn.Module):
    
    def __init__(self, w_num):
        super(Model, self).__init__()
        self.weights = Parameter(torch.zeros(w_num, 1), 
                                 requires_grad=True)
        self.weights_n = w_num
        self.init_weights()
                    
    def forward(self, x):
        net_input = x.mm(self.weights)
        return net_input

    def init_weights(self):
        self.weights = Parameter(torch.unsqueeze(torch.Tensor(xavier_init(self.weights_n)), 1), 
                         requires_grad=True)

model = Model(len(x[0]))
criterion = torch.nn.MSELoss()
optimizer_fcn = torch.optim.SGD

max_loss = 10e-13

n_i = 10000

def training(model, optimizer, x, y, max_loss, max_iter, loss_div = 10e6):
  
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

while True:
    model.init_weights()
    optimizer = optimizer_fcn(model.parameters(), lr=l_r)
    try:
        loss_v, weights = training(model, optimizer, x, y, max_loss, n_i)
        break
    except Exception as inst:
        print(inst)
        l_r -= lr_step
        if l_r < min_lr:
            raise Exception('Divergation for all learning rates.')
        print('-------------------------------------------\n')
        print('Starting with learning rate: {0}\n'.format(l_r))
        continue



outputs_t = model(x_t)
loss_t = criterion(outputs_t, y_t)

print('Error on training data: ', loss_v[-1])
print('Error on test data: ', loss_t.data.numpy()[0])
plot_training(model, loss_v, ind = x, outd = y.data.numpy()) #, w = np.array(weights))
plot_test(model, x_t, y_t.data.numpy())

plt.show()