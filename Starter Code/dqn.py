from collections import deque
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
import random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(
    *args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(
                np.float32(state)).unsqueeze(0), requires_grad=True)
            action_values = self(state)
            action = torch.argmax(action_values).item()
            action = (int)(action)
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = torch.Tensor.cpu(
        replay_buffer.sample(batch_size))

    state = Variable(torch.FloatTensor(
        np.float32(state)).squeeze(1), requires_grad=True)
    next_state = Variable(torch.FloatTensor(
        np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    y = reward + (gamma * model.forward(next_state))
    target = target_model.forward(state)
    # implement the loss function here
    loss = nn.functional.mse_loss(y, target)

    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        e = (state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        experiences = random.sample(self.buffer, batch_size)

        memory = np.array(experiences)

        state = torch.from_numpy(np.array(memory[:][0]))
        action = torch.from_numpy(np.array(memory[:][1]))
        reward = torch.from_numpy(np.array(memory[:][2]))
        next_state = torch.from_numpy(np.array(memory[:][3]))
        done = torch.from_numpy(np.array(memory[:][4]))

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
