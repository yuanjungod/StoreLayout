"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import torch.optim as optim

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 200
# N_ACTIONS = env.action_space.n
N_ACTIONS = 10
# N_STATES = env.observation_space.shape[0]
N_STATES = 10
print(N_ACTIONS, N_STATES)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class Rnn(nn.Module):
    def __init__(self, hidden_dim, n_layer):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(N_STATES, hidden_dim, n_layer,
                            batch_first=False)
        self.classifier = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # print(x)
        # x = torch.unsqueeze(torch.FloatTensor([x]), 0)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Rnn(10, 1), Rnn(10, 1)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):

        if np.random.uniform() < EPSILON:  # greedy
            x = torch.unsqueeze(torch.FloatTensor([x]), 0)
            actions_value = self.eval_net.forward(x)
            # print("actions_value", actions_value)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # print("b_s", b_s)
        b_s = b_s.view(-1, 1, 5)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        b_s_ = b_s_.view(-1, 1, 5)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Env(object):

    def __init__(self):
        self.stat = [i for i in range(10)]

    def reset(self):
        # self.stat = [-1 for i in range(N_ACTIONS)]
        self.stat = [i for i in range(10)]
        return self.stat

    def step(self, action):
        reward = 1 if action == self.stat[-1]+1 else 0
        return self.stat.append(action)[1:], reward, reward == 1


if __name__ == "__main__":
    dqn = DQN()
    env = Env()
    i_episode = 0
    while True:
        s = env.reset()
        ep_r = 0
        while True:
            a = dqn.choose_action(s)

            # take action
            s_, r, done = env.step(a)
            # print(s_)

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done and i_episode % 100 == 0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))
                    if ep_r > 0:
                        print(s_)

            if done:
                break
            s = s_
            i_episode += 1
