
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torch



x = Variable(torch.Tensor([[1.0, 1.0], 
                           [1.0, 2.1], 
                           [1.0, 3.6], 
                           [1.0, 4.2], 
                           [1.0, 6.0], 
                           [1.0, 7.0]]))
y = Variable(torch.Tensor([1.0, 2.1, 3.6, 4.2, 6.0, 7.0]))
weights = Variable(torch.zeros(2, 1), requires_grad=True)



# loss1 = []
# for i in range(5000):

#     net_input = x.mm(weights)
#     loss = torch.mean((net_input - y)**2)
#     loss.backward()
    
#     weights.data.add_(-0.0001 * weights.grad.data)
    
#     loss1.append(loss.data[0])
#     print('n_iter', i, 'loss', loss.data[0])
    
#     weights.grad.data.zero_()

# plt.plot(loss1)
# plt.show()



class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.weights = Parameter(torch.zeros(2, 1), 
                                 requires_grad=True)
    
    def forward(self, x):
        net_input = x.mm(self.weights)
        return net_input
        
model = Model()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss2 = []

for i in range(5000):
    optimizer.zero_grad()
    outputs = model(x)
    
    loss = criterion(outputs, y)
    loss2.append(loss.data[0])
    loss.backward()        

    optimizer.step()
    
plt.plot(range(5000), loss2)
plt.show()