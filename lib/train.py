from lib.data_loader import *
from lib.rnn import *
from lib.config import *


class TrainJob(object):

    def __init__(self):
        self.rnn = Rnn(IN_DIME, HIDDEN_DIM, N_LAYER, OUT_DIM)
        self.data_iter = DataLoader.data_generate1(50)
        self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=0.1)
        # self.loss_func = nn.MSELoss()
        # self.loss_func = nn.L1Loss()
        self.loss_func = torch.nn.CrossEntropyLoss()

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = LR * (0.95 ** (epoch // 10000))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, epoch_count):

        for epoch in range(epoch_count):
            train_x, train_y = next(self.data_iter)
            pred = self.rnn(train_x)

            self.optimizer.zero_grad()
            loss = self.loss_func(pred, train_y)
            loss.backward()
            self.optimizer.step()
            if epoch % 1000 == 0:  # 每50步，计算精度
                self.save_model()
                print(loss)

    def load_model(self, path):
        self.rnn.load_state_dict(torch.load(path))

    def save_model(self):
        torch.save(self.rnn.state_dict(), './rnn.pth')


if __name__ == "__main__":
    job = TrainJob()
    # job.load_model("./rnn.pth")
    job.train(500000)
    result = list()
    test_x = [4., 4., 4., 5., 5., 5., 5., 5., 5., 5.]
    # num_list = [0, 3, 3, 2, 2, 0, 0, 0, 0, 0]

    for i in range(30):
        # num_tensor = torch.FloatTensor(num_list)
        train_x = torch.unsqueeze(torch.FloatTensor(test_x), 0)
        train_x = train_x.view(-1, 1, IN_DIME)
        actions_value = job.rnn(train_x)
        prob = torch.nn.functional.softmax(actions_value, dim=1)
        # pred_cls = torch.argmax(prob, dim=1)

        print(prob, prob.sum())
        # actions_value = prob*num_tensor
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]
        # num_list[action] -= 1
        test_x.append(action)
        test_x = test_x[-10:]
        result.append(action)
    print(result)
