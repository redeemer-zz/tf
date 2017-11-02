
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import numpy as np
from tf_honu import *



time = np.arange(0,100,0.1, dtype=np.float32)
data = np.sin(time)
# data += np.random.uniform(-0.01, 0.01, len(time))
data[540] += 0.3
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
        self.weights = Parameter(torch.zeros(w_num, 1), 
                                 requires_grad=True)
    
    def forward(self, x):
        net_input = x.mm(self.weights)
        return net_input
        
model = Model(len(x[0]))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss2 = []


train = plt.figure()
plt.plot(y_train)
n_i = 1000
for i in range(n_i):
    optimizer.zero_grad()
    outputs = model(x)
    
    loss = criterion(outputs, y)
    loss2.append(loss.data[0])
    loss.backward()        
    optimizer.step()

    if i%10 == 0:
        plt.plot(outputs.data.numpy(), 'r', alpha=i/n_i)
    
outputs_t = model(x_t)
loss_t = criterion(outputs_t, y_t)
    

print(loss_t)
plt.figure()
plt.plot(loss2)
plt.figure()
plt.plot(y_test)
plt.plot(outputs_t.data.numpy(), 'r')

plt.show()