import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np


matrix_list = list()
label_list = list()
for i in range(20):
    matrix = [i+j for j in range(10)]
    matrix_list.append(matrix)
    label_list.append([i+10])

x_train = torch.from_numpy(np.array(matrix_list, dtype=np.float32))

y_train = torch.from_numpy(np.array(label_list, dtype=np.float32))


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(10, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# 开始训练
num_epochs = 100000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(inputs)
    print(out)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    if (epoch+1) % 20 == 0:
        # print(loss)
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, loss.data[0]))