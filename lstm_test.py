import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, out_dim):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # print(x)
        # x = torch.unsqueeze(torch.FloatTensor([x]), 0)
        out, _ = self.lstm(x)
        # print(out)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.5 * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    net = Rnn(10, 10, 1, 8)
    # net.load_state_dict(torch.load('linear.pth'))
    matrix_list = list()
    label_list = list()
    # for i in range(10):
    #     matrix = [i+j for j in range(10)]
    #     matrix = np.array(matrix).reshape(10, 1)
    #     matrix_list.append(matrix.tolist())
    #     label_list.append(i+10)
    matrix_list = [[i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9] for i in range(8)]
    label_list = [i for i in range(8)]
    x, y = matrix_list, label_list
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    # y = torch.unsqueeze(torch.LongTensor(y), 0)
    y = torch.LongTensor(y)
    x = x.view(-1, 1, 10)
    testdata_iter = iter([(i, j) for i, j in zip(matrix_list, label_list)])
    for epoch in range(100000):  # 数据集只迭代一次

        # y = y.view(-1, 1)
        # print(x.view(-1, 1, 10))
        # print(x)
        # print(y)
        pred = net(x)
        # print(pred)
        # print(y)
        # exit()
        lr = 0.1 * (0.95 ** (epoch // 6000))
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        # loss_func = nn.MSELoss()
        # loss_func = nn.L1Loss()
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        # print(loss)
        if epoch % 1000 == 0:  # 每50步，计算精度
            print(loss)
            print(lr)
            print(y)
            print(pred)
            print(net(torch.unsqueeze(torch.FloatTensor([[1, 1, 2, 3, 4, 5, 6, 7, 8, 10]]), 0)))
        #     with torch.no_grad():
        #         test_pred = net(x.view(-1, 1, 10))
        #         acc = (test_pred - y).sum().numpy() / test_pred.size()[0]
        #         print(f"{epoch}: accuracy:{acc}")
        #         # print(test_pred)
        #         print(lr)
        #         torch.save(net.state_dict(), './linear.pth')


