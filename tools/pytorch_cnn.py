import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import random

# Hyper Parameters
EPOCH = 10000  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.01  # learning rate
Category = 20


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=128,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                padding=0,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.Softmax(),  # activation
            nn.BatchNorm2d(128),
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(128, 64, 3, 2, 0),  # output shape (32, 14, 14)

            nn.Softmax(),  # activation
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(64, 32, 3, 2, 0),  # output shape (32, 14, 14)

            nn.Softmax(),  # activation
            nn.BatchNorm2d(32),
        )
        self.conv4 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(32, 16, 2, 2, 0),  # output shape (32, 14, 14)

            nn.Softmax(),  # activation
            nn.BatchNorm2d(16),
        )
        self.out = nn.Linear(in_features=16, out_features=Category, bias=False)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print("shape", x.shape)
        output = self.out(x)
        return output, x  # return x for visualization


seed_list = random.choice(range(Category))


def create(batch_size):
    map_list = [[seed_list for j in range(28*28)] for i in range(batch_size)]
    label_list = list()
    for i in range(len(map_list)):
        x = random.choice(range(26))
        y = random.choice(range(26))
        map_list[i][10] = -1
        map_list[i][10] = i % Category
        label_list.append(i % Category)

    train_x = torch.FloatTensor(map_list)
    train_x = train_x.view(-1, 1, 28, 28)
    # print(train_x)

    train_y = torch.LongTensor(label_list)
    # print(train_y)
    return train_x, train_y


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

for epoch in range(EPOCH):
    train_x,  train_y = create(50)
    output = cnn(train_x)[0]  # cnn output
    loss = loss_func(output, train_y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    pred_y = torch.max(output, 1)[1].data.squeeze().numpy()
    # print(pred_y)
    accuracy = float((pred_y == train_y.data.numpy()).astype(int).sum()) / float(train_y.size(0))
    print(accuracy)
