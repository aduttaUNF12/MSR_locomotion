import math
from datetime import date
import tqdm

from controller import Supervisor
import random
import struct
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple


__version__ = '08.07.21'


# Constants
EPSILON = 1
ALPHA = 0.1
MIN_EPSILON = 0.1
T = 0.1
BATCH_SIZE = 128
MAX_EPISODE = 5000
# MEMORY_CAPACITY = 10**10
MEMORY_CAPACITY = 2**30
# MEMORY_CAPACITY = 2**20
# MAX_EPISODE = 10
UPDATE_PERIOD = 10
EPISODE = 0
DATE_TODAY = date.today()

TOTAL_ELAPSED_TIME = 0
LEADER_ID = 1
TRIAL_PATH = ""
TRIAL_NAME = ""

NUM_MODULES = 3
INITIAL = [0, 0.5, 0]
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BASE_LOGS_FOLDER = None


# FROM MFRL paper
class ReplayBuffer:
    """a circular queue based on numpy array, supporting batch put and batch get"""
    def __init__(self, shape, dtype=np.float32):
        self.buffer = np.empty(shape=shape, dtype=dtype)
        self.head   = 0
        self.capacity   = len(self.buffer)
        self.return_buffer_len = 0

    def put(self, data):
        """put data to
        Parameters
        ----------
        data: numpy array
            data to add
        """
        head = self.head
        # n = len(data)
        n = data.size
        if head + n <= self.capacity:
            self.buffer[head:head+n] = data
            self.head = (self.head + n) % self.capacity
        else:
            split = self.capacity - head
            self.buffer[head:] = data[:split]
            self.buffer[:n - split] = data[split:]
            self.head = split
        return n

    def get(self, index):
        """get items
        Parameters
        ----------
        index: int or numpy array
            it can be any numpy supported index
        """
        return self.buffer[index]

    def clear(self):
        """clear replay buffer"""
        self.head = 0


replay_buf_reward = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
replay_buf_action = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.int32)
replay_buf_state = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
replay_buf_state_ = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)


class Action:
    def __init__(self, name, function):
        self.name = name
        self.func = function


# the NN itself
class CNN(nn.Module):
    def __init__(self, number_of_modules, lr, n_actions):
        super(CNN, self).__init__()
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        # self.conv1 = nn.Conv2d(1, n_actions, kernel_size=3).to(self.device)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)

        # self.conv2 = nn.Conv2d(n_actions, number_of_modules*n_actions, kernel_size=3).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1).to(self.device)
        self.bn2 = nn.BatchNorm2d(64).to(self.device)

        # self.conv3 = nn.Conv2d(number_of_modules*n_actions, number_of_modules*n_actions*5, kernel_size=3).to(self.device)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=1).to(self.device)
        self.conv3 = nn.Conv2d(64, BATCH_SIZE, kernel_size=1).to(self.device)
        self.bn3 = nn.BatchNorm2d(128).to(self.device)

        # self.fc1 = nn.Linear(1, number_of_modules*n_actions).to(self.device)

        # x = torch.randn(BATCH_SIZE, 1).to(self.device).view(-1, 1, BATCH_SIZE, 1)
        x = torch.rand((BATCH_SIZE, 1)).to(self.device).view(-1, 1, BATCH_SIZE, 1)
        # x = torch.randn(1, 2, BATCH_SIZE, 1).to(self.device)
        # x = torch.rand((BATCH_SIZE, 1)).to(self.device).view(-1, )
        # print(f"x torch rand: {x}")

        self._to_linear = None
        self.convs(x)

        # self.fc1 = nn.Linear(self._to_linear, 515).to(self.device)
        self.fc1 = nn.Linear(self._to_linear, 515).to(self.device)
        self.fc2 = nn.Linear(515, 3).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def convs(self, x):
        # print(f"!!!!! INSIDE CONVS with x: {x}\nshape x: {x.shape}")
        # x = self.conv1(x)
        # print(f"conv1 res: {x}")
        # x = F.relu(x)
        # print(f"relu res: {x}")
        # x = F.max_pool2d(x, (1, 1))
        # print(f"max_pool2d res: {x}")
        # exit(1)
        # x = F.max_pool2d(F.relu(self.conv1(x)), (1, 1))


        # print(f"init x: {x} (size) {x.size()}")
        # exit(123)
        x = self.conv1(x)
        # print(f"after conv1: {x}")
        x = self.bn1(x)
        # print(f"after bn1: {x}")
        x = F.relu(x)
        # print(f"after relu: {x}")
        # print("######################################################")
        x = self.conv2(x)
        # print(f"after conv2: {x}")
        x = self.bn2(x)
        # print(f"after bn2: {x}")
        x = F.relu(x)
        # print(f"after relu2: {x}")
        # print("######################################################")
        x = self.conv3(x)
        # print(f"after conv3: {x}")
        x = self.bn3(x)
        # print(f"after bn3: {x}")
        x = F.relu(x)
        # print(f"after relu3: {x} (size): {x.size()}  (shape): {x.shape}")
        # print("######################################################")

        # print(f"Past conv1 x: {x}")
        # x = F.max_pool2d(F.relu(self.conv2(x)), (1, 1))
        # print(f"Past conv2 x: {x}")
        # x = F.max_pool2d(F.relu(self.conv3(x)), (1, 1))
        # print(f"Past conv3 x: {x}")

        if self._to_linear is None:
            # print(f"init _to_linear: {self._to_linear}")
            # print(f"x size {x.size()}")
            #
            # print(f"x[0]: {x[0]}")
            # print(f"x[0]: {x[0].shape[0]}")
            # print(f"x[0]: {x[0].shape[0]*x[0]}")
            # print(f"x[0]: {x[0].shape[0]*x[0].shape[1]}")
            # print(f"x[0]: {x[0].shape[0]*x[0].shape[1]*x[0]}")
            #
            #
            # print(f"x[0]: {x[0].shape[0]*x[0].shape[1]*x[0].shape[2]}")
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            # exit(22)
            # print(f"_to_linear: {self._to_linear}\nx: {x}")
        return x

    def forward(self, actions):
        x = self.convs(actions)
        # print(f"x size(0): {x.size(0)}")
        x = x.view(x.size(0), -1)
        # print(f"x after view: {x}")
        x = self.fc1(x)
        # print(f"x after fc1: {x} (size) {x.size()}")
        x = F.relu(x)
        # print(f"x after relu(post fc1): {x}")
        x = self.fc2(x)
        # print(f"x after fc2: {x}")
        # return x
        # exit(66)
        # print(f"self._to_linear: {self._to_linear}")
        # print(f"x.view(): {x.view(1,  515)}")
        # exit(13)
        # x = x.view(-1, self._to_linear)
        # print(f"x after x.view: {x}")
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # print(f"F.softmax(x): {F.softmax(x, dim=1)}")
        # print(f"x: {x}")
        # exit(1)
        # return F.softmax(x, dim=1)
        return x[0]

        #
        #
        # # unsqueeze is needed to turn 1-d input into a 4-d input
        # # print(f"Actions passed to NN: {actions}")
        # actions = actions.unsqueeze(1)
        # # print(f"Actions after unsqueeze(1): {actions}")
        # actions = actions.unsqueeze(1)
        # # print(f"Actions after unsqueeze(1): {actions}")
        # actions = actions.unsqueeze(1)
        # # print(f"Actions after unsqueeze(1): {actions}")
        # # print(f"Actions passed to fc1: {self.fc1(actions)}")
        # s1 = self.conv1(actions)
        # # print(f"s1: {s1}")
        # # layer1 = F.relu(self.fc1(actions))
        # # layer1 = F.relu(self.fc1(actions))
        # layer1 = F.relu(s1)
        # # print(f"Layer 1: {layer1}")
        # s2 = self.conv2(layer1)
        # # print(f"s2: {s2}")
        # # layer2 = F.relu(self.fc2(layer1))
        # layer2 = F.relu(s2)
        # # print(f"Layer 2: {layer2}")
        # layer2 = layer2.view(-1, self.number_of_modules * self.n_actions)
        # # print(f"Layer 2:{ layer2}")
        # s3 = self.fc1(layer2)
        # # print(f"s3: {s3}")
        # # return self.fc3(layer2)
        # return s3[0]


policy_net = CNN(NUM_MODULES, 0.001, 3)
# agent who controls NN, in this case the main module (module 1)


class Agent:
    # policy_net = does all of the training and tests
    # target_net = final network

    def __init__(self, module_number, number_of_modules, n_actions, lr, alpha=0.99,
                 epsilon=1, eps_dec=1e-5, eps_min=0.01):
        self.module_number = module_number   # only used for logging
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.updated = False

        self.action_space = [i for i in range(self.n_actions)]

        # q estimate
        self.target_net = CNN(self.number_of_modules, self.lr, n_actions)
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()

    # Greedy action selection
    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):
            temp = torch.full((BATCH_SIZE, ), module_actions, dtype=torch.float).view(-1, 1, BATCH_SIZE, 1).to(policy_net.device)
            action = policy_net.forward(temp)
            return [np.argmax(action.to('cpu').detach().numpy())]
        else:
            action = np.random.choice(self.action_space, 1)
        return action

    def decrement_epsilon(self):
        # decrease epsilon by set value if epsilon != min epsilon
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        # self.main_network.optimizer.zero_grad()
        policy_net.optimizer.zero_grad()
        # TODO: priority memory
        # pull a random BATCH of data from replay buffer
        random_sample = np.random.randint(1, replay_buf_state.head, BATCH_SIZE, np.int)
        # a matrix of sample locations in a replay buffer which are than being referenced once pulling needed data types
        # t = np.ones(shape=(BATCH_SIZE,), dtype=np.int)
        # prev_random_sample = np.subtract(random_sample, t)

        # .view(-1, 1, BATCH_SIZE, 1) is used to reshape the tensor to fit the expected input,
        # it converts size [128] (original tensor size) to [1, 1, 128, 1]
        batch_states = torch.tensor(replay_buf_state.get(random_sample), dtype=torch.float)
        batch_action = torch.tensor(replay_buf_action.get(random_sample), dtype=torch.int64)
        # batch_reward = torch.tensor(replay_buf_reward.get(random_sample)).to(self.target_net.device).view(-1, 1, BATCH_SIZE, 1)
        batch_reward = torch.tensor(replay_buf_reward.get(random_sample)).to(self.target_net.device)
        # batch_reward = torch.tensor(replay_buf_reward.get(random_sample))
        # batch_states_next = torch.tensor(replay_buf_state_.get(random_sample)).to(self.target_net.device)
        batch_states_next = torch.tensor(replay_buf_state_.get(random_sample))

        state_action_values = []
        for index, state in enumerate(batch_states):
            # creates a [BATCH_SIZE] shaped tensor full of state values
            full_states = torch.full((BATCH_SIZE, ), state, dtype=torch.float).view(-1, 1, BATCH_SIZE, 1).to(policy_net.device)
            # r = policy_net(full_states).to('cpu')
            r = policy_net(full_states)
            # r = r.to('cpu')
            state_action_values.append(r[batch_action[index]])
        del full_states
        state_action_values = torch.stack(state_action_values)

        expected_state_action_values = []
        for index, state_ in enumerate(batch_states_next):
            full_states_ = torch.full((BATCH_SIZE, ), state_, dtype=torch.float).view(-1, 1, BATCH_SIZE, 1).to(self.target_net.device)
            r = self.target_net(full_states_)
            r = torch.tensor(np.argmax(r.to('cpu').detach().numpy()), dtype=torch.float).to(self.target_net.device)
            expected_state_action_values.append(r)
        del r, full_states_

        expected_state_action_values = (torch.stack(expected_state_action_values) * self.alpha) + batch_reward
        # expected_state_action_values = (torch.stack(expected_state_action_values) * self.alpha)
        # expected_state_action_values_fin = []
        # # had to do the loop since after addition of reward the shape gets messed up
        # for index, esav in enumerate(expected_state_action_values):
        #     expected_state_action_values_fin.append(esav + batch_reward[index])

        # expected_state_action_values_fin = torch.stack(expected_state_action_values_fin)
        # expected_state_action_values_fin = expected_state_action_values_fin.float().to('cpu')

        # expected_state_action_values = expected_state_action_values.float()
        # expected_state_action_values = expected_state_action_values.to('cpu')


        # loss = policy_net.loss(state_action_values, expected_state_action_values).to(policy_net.device)
        # loss = policy_net.loss(state_action_values, expected_state_action_values_fin)
        state_action_values = state_action_values.float()
        expected_state_action_values = expected_state_action_values.float()
        loss = policy_net.loss(state_action_values, expected_state_action_values)
        # print(f"loss: {loss}")
        loss.backward()
        policy_net.optimizer.step()
        self.decrement_epsilon()

        if EPISODE % UPDATE_PERIOD == 0 and self.updated is False:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{EPISODE}")
            self.target_net.load_state_dict(policy_net.state_dict())

            if BASE_LOGS_FOLDER is not None:
                torch.save(module.agent.target_net.state_dict(), os.path.join(BASE_LOGS_FOLDER, "agent.pt"))

            self.updated = True
        elif EPISODE % UPDATE_PERIOD != 0:
            self.updated = False

        return loss

        # # loss = self.main_network.loss(state_action_q_values, expected_next_state_q_values).to(self.main_network.device)
        # loss = train_network.loss(state_action_q_values, expected_next_state_q_values).to(train_network.device)
        # # print(f"loss: {loss}")
        # loss.backward()
        # # self.main_network.optimizer.step()
        # train_network.optimizer.step()
        # self.decrement_epsilon()
        #
        #
        #
        # exit(111)
        #
        #
        # # batch_states = torch.tensor(replay_buf_state.get(random_sample), dtype=torch.float).to(policy_net.device).view(-1, 1, BATCH_SIZE, 1)
        # # print(f"batch[0] : {batch_states[0]}")
        # # exit(11)
        # batch_action = torch.tensor(replay_buf_action.get(random_sample)).to(policy_net.device).view(-1, 1, BATCH_SIZE, 1)
        # batch_reward = torch.tensor(replay_buf_reward.get(random_sample)).to(policy_net.device).view(-1, 1, BATCH_SIZE, 1)
        #
        # # x = torch.randn(BATCH_SIZE, 1).view(-1, 1, BATCH_SIZE, 1).to(self.device)
        # # print(f"torch.randn(Batch_size,1): {torch.randn(BATCH_SIZE, 1)} (size) {torch.randn(BATCH_SIZE, 1).size()}")
        # # print(f"batch_states resized: {batch_states.view(-1, 1, batch_states.size()[0], 1).size()}")
        # # print(f"batch_states: {batch_states} (size) {batch_states.size()}")
        # # exit(11)
        # # state_action_values = policy_net(batch_states).gather(1, batch_action)
        # # state_action_values = policy_net(batch_states).
        # # print(f"state_action_values: {state_action_values}")
        #
        #
        #
        # exit(11)
        # # states__t = torch.tensor(state_, dtype=torch.float).to(self.main_network.device)
        # # states__t = torch.tensor(state_, dtype=torch.float).to(train_network.device)
        #
        # # # batch of past states_
        # # batch_state_ = replay_buf_state_.get(random_sample)
        # # # batch of states
        # # batch_sates = replay_buf_state.get(random_sample)
        # # # batch of actions
        # # batch_action = replay_buf_action.get(random_sample)
        # # batch_action_ = replay_buf_action.get(prev_random_sample)
        # # # batch of rewards
        # # batch_reward = replay_buf_reward.get(random_sample)
        # # batch_reward_ = replay_buf_reward.get(prev_random_sample)
        # # # state  + action  + reward   training batch
        # # # state_ + action_ + reward_  prev_batch
        # #
        # # print(f"batch_states: {batch_sates}")
        # # print(f"batch_action: {batch_action}")
        # # print(f"batch_rewards: {batch_reward}")
        # # s_a_a = np.array(list(zip(batch_sates, batch_action, batch_reward)), dtype=np.float)
        # # # s_a_a = np.array(list(zip(batch_sates, batch_action)), dtype=np.float)
        # # # shape = [128, 3] need [32,1,1,1]
        # # s_a_a_t = torch.tensor(s_a_a, dtype=torch.float).to(self.target_net.device)
        # # print(f"f: {s_a_a_t}")
        # # # shape = [128, 1, 3] need [32,1,1,1]
        # # s_a_a_t = s_a_a_t.unsqueeze(1)
        # # # shape = [128, 1, 1, 3] works with [32, 1, 1, 1]
        # # s_a_a_t = s_a_a_t.unsqueeze(1)
        # # r = self.target_net.forward(s_a_a_t)
        # # print(f"r: {r}")
        # #
        # # v_a_a = np.array(list(zip(batch_state_, batch_action_, batch_reward_)), dtype=np.float)
        # # # v_a_a = np.array(list(zip(batch_state_, batch_action_, batch_reward_)), dtype=np.float)
        # # print(f"v_a_a: {v_a_a}")
        # # v_a_a_t = torch.tensor(v_a_a, dtype=torch.float).to(train_network.device)
        # # print(f"v_a_a_t: {v_a_a_t}")
        # #
        # # v_a_a_t = v_a_a_t.unsqueeze(1)
        # # print(f"v_a_a_t us1: {v_a_a_t}")
        # #
        # # v_a_a_t = v_a_a_t.unsqueeze(1)
        # # print(f"v_a_a_t us2: {v_a_a_t}")
        # #
        # # g = train_network.forward(v_a_a_t)
        # # print(f"g: {g}")
        # #
        # # g = torch.max(g).to(train_network.device) * self.alpha
        # # print(f"g 2: {g}")
        # # exit(11)
        #
        #
        #
        #
        # # s_t_tensor = self.Q.forward(torch.transpose(torch.tensor(batch_sates, dtype=torch.float).to(self.Q.device), 0, -1))
        # # a_tensor = torch.tensor(batch_action, dtype=torch.int64).to(self.Q.device)
        # # state_action_q_values.append(torch.gather(s_t_tensor, 0, a_tensor))
        # for s_t, a in zip(batch_sates, batch_action):
        #     s_t_tensor = self.target_net.forward(torch.tensor([s_t], dtype=torch.float).to(self.target_net.device))
        #     # s_t_tensor = train_network.forward(torch.tensor([s_t], dtype=torch.float).to(train_network.device))
        #     a_tensor = torch.tensor([a], dtype=torch.int64).to(self.target_net.device)
        #     # a_tensor = torch.tensor([a], dtype=torch.int64).to(train_network.device)
        #     # print(f"s_t_tensor: {s_t_tensor}")
        #     # print(f"a_tensor: {a_tensor}")
        #     state_action_q_values.append(torch.gather(s_t_tensor, 0, a_tensor))
        #
        # #  get q value for state_
        # # for s_q_id, s_q in enumerate(states__t):
        # #     if batch_reward[s_q_id] is None:
        # #         r = batch_reward[s_q_id]
        # #     else:
        # #         r = 0
        # #     expected_next_state_q_values.append(
        # #         (torch.max(train_network(torch.tensor([s_q], dtype=torch.float).to(self.main_network.device))) * self.alpha) + r)
        # #         # (torch.max(self.Q.forward(torch.tensor([s_q], dtype=torch.float).to(self.Q.device))) * self.alpha) + r)
        #
        # # turn tensor lists into a single tensor
        # state_action_q_values = torch.stack(state_action_q_values)
        # expected_next_state_q_values = torch.stack(expected_next_state_q_values)
        #
        # # loss = self.main_network.loss(state_action_q_values, expected_next_state_q_values).to(self.main_network.device)
        # loss = train_network.loss(state_action_q_values, expected_next_state_q_values).to(train_network.device)
        # # print(f"loss: {loss}")
        # loss.backward()
        # # self.main_network.optimizer.step()
        # train_network.optimizer.step()
        # self.decrement_epsilon()
        #
        # if EPISODE % UPDATE_PERIOD == 0 and self.updated is False:
        #     print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{EPISODE}")
        #     self.target_net.load_state_dict(train_network.state_dict())
        #
        #     if BASE_LOGS_FOLDER is not None:
        #         torch.save(module.agent.target_net.state_dict(), os.path.join(BASE_LOGS_FOLDER, "agent.pt"))
        #
        #     self.updated = True
        # elif EPISODE % UPDATE_PERIOD != 0:
        #     self.updated = False
        #
        # return loss


# robot module instance
class Module(Supervisor):
    def __init__(self):
        # TODO: add mean values to the buffer
        # TODO: CNN instead of DQN
        Supervisor.__init__(self)
        #  webots section
        self.bot_id = int(self.getName()[7])
        self.timeStep = int(self.getBasicTimeStep())
        self.prev_episode = EPISODE
        self.episode_reward = 0
        self.self_message = bytes

        self.gps = self.getDevice("gps")
        self.gps.resolution = 0
        self.gps.enable(self.timeStep)

        self.emitter = self.getDevice("emitter")
        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.timeStep)

        self.motor = self.getDevice("motor")

        # making all modules have NN
        self.agent = Agent(self.bot_id, NUM_MODULES, 3, 0.001, alpha=ALPHA,
                           epsilon=EPSILON, eps_dec=0.001, eps_min=MIN_EPSILON)
        self.receiver.setChannel(0)
        self.emitter.setChannel(0)

        self.actions = [
            Action("Up", self.action_up),
            Action("Down", self.action_down),
            Action("Neutral", self.action_neutral)
        ]

        self.old_pos = self.gps.getValues()
        self.new_pos = self.old_pos
        self.t = 0
        self.batch_ticks = 0
        self.reward = 0
        self.loss = None
        self.calc_reward()

        self.global_actions = [1]*NUM_MODULES
        self.mean_action = 0
        self.prev_mean_action = 0
        """
        states:  0 - min
                 1 - zero
                 2 - max
        """
        # TODO: it is a collection of all of the module states, but later will have to be only for neighbors
        self.current_state = 1
        self.prev_state = self.current_state
        self.act = None
        # self.previous_module_state = self.global_actions

        # getting leader to decide initial actions
        self.current_action = np.random.choice([i for i in range(3)], 1)[0]
        # print(f"[*] {self.bot_id} - initial action: {self.current_action}")
        self.prev_actions = self.current_action

        # TEMP FOR ERROR CHECKING TODO
        self.tries = 0


        self.state_changer()
        self.notify_of_an_action(self.current_action)

    # notify other modules of the chosen action
    def notify_of_an_action(self, state):
        message = struct.pack("i", self.bot_id)
        message += struct.pack("i", state)

        self.global_actions[self.bot_id-1] = state
        ret = self.emitter.send(message)
        if ret == 1:
            pass
        elif ret == 0:
            print(f"[*] {self.bot_id} - acts were not sent")
            exit(3)
        else:
            print(f"[*] {self.bot_id} - error while sending data")
            exit(123)

        self.self_message = message

    # gets actions which were sent by the leader
    def get_other_module_actions(self):
        if self.receiver.getQueueLength() > 0:
            while self.receiver.getQueueLength() > 0:
                message = self.receiver.getData()
                data = struct.unpack("i" + "i", message)
                self.global_actions[data[0]-1] = data[1]
                self.receiver.nextPacket()
        else:
            print("Error: no packets in receiver queue.")
            return 2

    def action_up(self):
        self.motor.setPosition(1.57)

    def action_down(self):
        self.motor.setPosition(-1.57)

    # no action is taken, module just passes
    def action_neutral(self):
        # TODO: REMOVE IF HAS ODD RESULTS
        # experiment to see if making neutral not do anything will be beneficial or not


        # self.motor.setPosition(0)
        pass

    # calculates reward based on the travel distance from old position to the new, calculates hypotenuse of two points
    def calc_reward(self):
        self.reward = math.hypot(self.old_pos[0]-self.new_pos[0], self.old_pos[2]-self.new_pos[2])
        # logger(bot_id=self.bot_id, reward=self.reward)

    # changes the state of the module based on the action.
    #   if the current state is down (0) and action is 0 (up) the state will be neutral (1)
    #   same logic backwards, is state is neutral (1) and action is 1 (down), the state will be down(0)
    #   if state is already up/down and we get action up/down module stays in current position
    def state_changer(self):
        self.prev_state = self.current_state

        if int(self.current_action) == 0:
            if int(self.current_state) < 2:
                self.current_state += 1

        elif int(self.current_action) == 1:
            if int(self.current_state) > 0:
                self.current_state -= 1

        elif int(self.current_action) == 2:
            pass

        else:
            print(f"[{self.bot_id}]  Error with state_changer !!!!!!!!!!", file=sys.stderr)
            exit(2)

    # Loops through NUM_MODULES+1 (since there is no module 0), sums actions of all modules except for current
    # divides by the total number of modules
    def calc_mean_action(self):
        a = 0
        for m in range(NUM_MODULES + 1):
            if m != self.bot_id:
                try:
                    a += self.global_actions[m]
                except IndexError:
                    pass
        # NUM_MODULE - 1 since we are taking a mean of all except current modules
        self.prev_mean_action = self.mean_action
        self.mean_action = a / (NUM_MODULES - 1)
        # logger(mean_action=self.mean_action, a=a)
        # self.mean_action_rounder()

    # Takes mean_action and rounds it down
    def mean_action_rounder(self):
        dec, n = math.modf(float(self.mean_action))
        if dec < 0.5:
            n = n
        else:
            n = n + 1
        self.mean_action = n

    def learn(self):
        # If current action is None
        if self.act is None:
            # get actions which were send by other modules, if ret is 2 that means that not all modules sent their
            # actions yet, so loop while you wait for them to come
            ret = self.get_other_module_actions()
            if ret == 2:
                if self.tries > 100:
                    print("Error with tries")
                    exit(11)
                self.tries += 1
                return
            # select an action corresponding to the current_action number
            self.act = self.actions[self.current_action]
            self.calc_mean_action()

        elif self.t + (self.timeStep / 1000.0) < T:
            # carry out a function
            self.act.func()

        elif self.t < T:
            # get new position
            self.new_pos = self.gps.getValues()
            # calculate the reward
            self.calc_reward()
            # if reward is missing, skip. There were errors when Webots would not properly assign GPS coordinates
            # causing reward to be none
            if self.reward != self.reward or self.reward is None:
                pass
            else:
                # pass values to corresponding replay buffers
                replay_buf_reward.put(np.array(self.reward))
                # replay_buf_state.put(np.array(self.current_state))
                replay_buf_state.put(np.array(self.mean_action))
                # replay_buf_state_.put(np.array(self.prev_state))
                replay_buf_state_.put(np.array(self.prev_mean_action))
                replay_buf_action.put(np.array(self.current_action))

                # for MFRL memory
                #     state will keep track of this
                replay_buf_state.return_buffer_len = \
                    min(MEMORY_CAPACITY, replay_buf_state.return_buffer_len + self.batch_ticks)
                # add reward to current episode_reward
                self.episode_reward += self.reward
                # If Episode changed
                if EPISODE > self.prev_episode:
                    if self.bot_id == LEADER_ID:
                        # print(f"Buffer len: {replay_buf_state.return_buffer_len} "
                        #       f"{(replay_buf_state.return_buffer_len/MEMORY_CAPACITY)*100}%"
                        #       f" Loss: {self.loss}")
                        # logger
                        writer(self.bot_id, NUM_MODULES, TOTAL_ELAPSED_TIME, self.episode_reward, self.loss)
                    self.episode_reward = 0
                    self.prev_episode = EPISODE
                # batch is full
                if self.batch_ticks > BATCH_SIZE:
                    # run the NN and collect loss
                    self.loss = self.agent.learn()
                    self.batch_ticks = 0
                else: self.batch_ticks += 1

            # TODO: change to neighbor modules later on
            # set previous action option to the new one
            self.prev_actions = self.current_action
            self.current_action = self.agent.choose_action(self.mean_action)[0]
            # run the action and change the state
            self.state_changer()
            # notify others of current action
            self.notify_of_an_action(self.current_action)

        else:
            self.act = None
            self.old_pos = self.gps.getValues()
            self.t = 0


# logs the information throughout the trial run, collects time_step, reward, loss, and Episode number
def writer(name, num_of_bots, time_step, reward, loss):
    global BASE_LOGS_FOLDER

    if BASE_LOGS_FOLDER is None:
        log_path = os.path.join(os.getcwd(), "LOGS")
        if not os.path.isdir(log_path):
            try:
                os.mkdir(log_path)
            except FileExistsError:
                pass

        set_folder_path = os.path.join(log_path, "{}_MODULES".format(num_of_bots))
        if not os.path.isdir(set_folder_path):
            try:
                os.mkdir(set_folder_path)
            except FileExistsError:
                pass

        current_run = os.path.join(set_folder_path, "{}_RUN".format(DATE_TODAY))
        if not os.path.isdir(current_run):
            try:
                os.mkdir(current_run)
            except FileExistsError:
                pass
        BASE_LOGS_FOLDER = current_run

    file_name = "{}_MODULES_{}.txt".format(num_of_bots, name)
    file_path = os.path.join(BASE_LOGS_FOLDER, file_name)
    with open(file_path, "a") as fin:
        fin.write('{},{},{},{}\n'.format(time_step, reward, loss, EPISODE))


def logger(**kwargs):
    with open("log.txt", "a") as fin:
        for kwarg in kwargs:
            fin.write("{}: {}      ".format(kwarg, kwargs[kwarg]))
        fin.write("\n")
        fin.write("=================== ENTRY END ========================\n")

    kl = len(kwargs)
    with open("log.csv", "a") as fin:
        for p, kwarg in enumerate(kwargs):
            if (p+1) >= kl:
                fin.write("{}".format(kwargs[kwarg]))
            else:
                fin.write("{},".format(kwargs[kwarg]))
        fin.write("\n")


if __name__ == '__main__':
    import time
    start_time = time.time()
    print(f"Starting the training: {time.strftime('%H:%M:%S', time.localtime())}")
    eps_history = []
    filename = "null"
    module = Module()
    assign_ = False
    learn = True
    i = 0
    while i < 100:
        i += 1
        time.sleep(0.05)
    print(f"Finished buffer period in: {time.time()-start_time} ===== {time.strftime('%H:%M:%S', time.localtime())}")

    last_episode = time.time()
    while module.step(module.timeStep) != -1:
        i = 0
        while i < 1000:
            i += 1

        module.learn()
        module.t += module.timeStep / 1000.0
        TOTAL_ELAPSED_TIME += module.timeStep / 1000.0
        if 0 <= TOTAL_ELAPSED_TIME % 60 <= 1:
            if not assign_:
                EPISODE += 1

                if module.bot_id == LEADER_ID:
                    print(f"Episode: {EPISODE} -- "
                          f"{time.time() - start_time} ===== time since last episode: {time.time() - last_episode} ====== Episode reward: {module.episode_reward} == Loss: {module.loss}")
                    # last_episode = time.time()

                assign_ = True
                module.simulationReset()
                module.old_pos = module.gps.getValues()
                # if module.bot_id == LEADER_ID:
                #     print(f"========================== M1 old_pos: {module.old_pos}")
                #     with open("M1_Rev_Track.txt", "a") as fin:
                #         fin.write("{},{},{}\n".format(time.time() - start_time,
                #                                       module.episode_reward, module.old_pos[0]))
                last_episode = time.time()
        else:
            # assign_ is a temp needed to prevent infinite loop on the first Episode
            assign_ = False
        if EPISODE > MAX_EPISODE:
        # if EPISODE > 150:
            end = time.time()
            print(f"Runtime: {end - start_time}")
            print("LOGGING OFF")
            exit()
