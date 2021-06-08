from datetime import date

from controller import Robot, Supervisor
import random
import struct
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
from collections import namedtuple

# sumTree
# SumTree
# a binary tree data structure where the parent’s value is the sum of its children

#  TODO: FINISH THIS PART
"""
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
"""

__version__ = '04.30.21'


# Constants
EPSILON = 1
ALPHA = 0.1
MIN_EPSILON = 0.1
T = 0.1
BATCH_SIZE = 128
MAX_EPISODE = 5000
MEMORY_CAPACITY = 10**10
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


class NNDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def get_random_line(self):
        total_bytes = os.stat(self.filename).st_size
        random_point = random.randint(0, total_bytes)
        file = open(self.filename)
        file.seek(random_point)
        file.readline() # skip this line to clear the partial line
        return file.readline()

    def get_batch(self):
        i = 0
        batch = []
        while i < BATCH_SIZE:
            # batch.append(self.get_random_line())
            line = self.get_random_line().split(',')
            fline = [x.replace('\n', '') for x in line[:4]]
            if "" not in fline:
                # print(f"Line: {line} ================= Line [:4]: {fline} ============ TOUPLE FLINE: {tuple(float(x) for x in fline)}")
                # batch.append(tuple(float(x.replace("\n", "")) for x in self.get_random_line().split(',')[:3]))
                batch.append(tuple(float(x) for x in fline))
                i += 1
        return batch

    def add_data(self, data):
        if self.__len__() > MEMORY_CAPACITY:
            pass
        else:
            with open(self.filename, "a") as fin:
                fin.write(data)

    def __len__(self):
        with open(self.filename, "r") as fout:
            count = sum(1 for _ in fout)
        return count

    def __iter__(self):
        file_iter = open(self.filename)
        return file_iter


# FROM MFRL paper
class ReplayBuffer:
    """a circular queue based on numpy array, supporting batch put and batch get"""
    def __init__(self, shape, dtype=np.float32):
        self.buffer = np.empty(shape=shape, dtype=dtype)
        self.head   = 0
        self.capacity   = len(self.buffer)

    def put(self, data):
        """put data to
        Parameters
        ----------
        data: numpy array
            data to add
        """
        head = self.head
        n = len(data)
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
replay_buf_mask = ReplayBuffer(shape=(MEMORY_CAPACITY,))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# memory = ReplayMemory(10**10)


class Action:
    def __init__(self, name, function):
        self.name = name
        self.func = function


# the NN itself
class LinearDQN(nn.Module):
    # inputs personal action, and action from each of the modules
    # number_of_modules = number of modules in the chain
    # lr = learning rate
    # n_actions = number of possible actions
    def __init__(self, number_of_modules, lr, n_actions):
        super(LinearDQN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        self.fc1 = nn.Linear(1, number_of_modules*n_actions).to(self.device)
        # add the current state of the modules
        self.fc2 = nn.Linear(number_of_modules*n_actions, number_of_modules*n_actions*10).to(self.device)
        self.fc3 = nn.Linear(number_of_modules * n_actions * 10, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, actions):
        layer1 = F.relu(self.fc1(actions))
        layer2 = F.relu(self.fc2(layer1))
        return self.fc3(layer2)


policy_net = LinearDQN(NUM_MODULES, 0.001, 3)
# target_net = LinearDQN(NUM_MODULES, 0.001, 3)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# agent who controls NN, in this case the main module (module 1)

# TODO: make a decorator for data logging
class Agent:
    def __init__(self, module_number, number_of_modules, n_actions, lr, database, alpha=0.99,
                 epsilon=1, eps_dec=1e-5, eps_min=0.01):
        self.module_number = module_number
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.database = NNDataset(database)

        self.action_space = [i for i in range(self.n_actions)]

        # q estimate
        self.Q = LinearDQN(self.number_of_modules, self.lr, n_actions)
        self.Q.load_state_dict(policy_net.state_dict())
        self.Q.eval()

    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):

            state = torch.tensor([module_actions], dtype=torch.float).to(self.Q.device)
            action = self.Q.forward(state)
            # getting a greedy vector of our actions
            # print(f"greedy actions: {torch.argmax(action)}")
            return [torch.argmax(action)]
        else:
            action = np.random.choice(self.action_space, 1)
            # print(f"random actions: {action}")
        return action

    def decrement_epsilon(self):
        # decrease epsilon by set value if epsilon != min epsilon
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        self.Q.optimizer.zero_grad()

        # TODO: priority memory

        transitions = self.database.get_batch()
        # if self.module_number == LEADER_ID:
        #     print(f"TRANSITIONS: {transitions}")
        #     print("===================================")
        #     # print(f"TRANSION CAST: {Transition(*transitions)}")
        #     t = zip(zip(*transitions))
        #     print(f"TRANSITIONS ZIP: {t.__next__()}")
        # transitions = memory.sample(BATCH_SIZE)
        # try:
        batch = Transition(*zip(*transitions))
        # except TypeError:
        #     pass
        # print(f"BATCH STATE: {[*batch.state]}")
        # exit(12)
        # batch = Transition(*transitions)

        # states_t = torch.tensor([*batch.state], dtype=torch.float).to(self.Q.device)
        states__t = torch.tensor([*batch.next_state], dtype=torch.float).to(self.Q.device)
        # print(f"states_t: {states_t}")
        # print(f"zip(state_t, batch.action): {zip(states_t, batch.action).__init__()}")
        # exit()
        # if LEADER_ID == self.module_number:
        #     # print(f"states_t: {states_t}")
        #     print(f"states_t: {batch.state}")
        #     print("===========================================")
        #     print(f"states__t: {states__t}")
        #     print("===========================================")
        #     print(f"actions: {batch.action}")
        #     print("===========================================")
        #     print(f"actions: {batch.reward}")
        #     print("===========================================")

        state_action_q_values = []
        # get q value for state
        # for s_t, a in zip(states_t, batch.action):
        for s_t, a in zip(batch.state, batch.action):
            # if LEADER_ID == self.module_number:
            #     print(f"s_t: {s_t} ===================== a: {a}")
            #     state_action_q_values.append(torch.gather(
            #         # policy_net(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)), 0, a))
            #         # self.Q.forward(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)), 0, a))
            #         self.Q.forward(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)),
            #         0, torch.tensor(a, dtype=torch.int64).to(self.Q.device)))
            #
            # exit()
            state_action_q_values.append(torch.gather(
                # policy_net(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)), 0, a))
                # self.Q.forward(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)), 0, a))
                self.Q.forward(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)),
                0, torch.tensor(a, dtype=torch.int64).to(self.Q.device)))

        expected_next_state_q_values = []
        #  get q value for state_
        for s_q_id, s_q in enumerate(states__t):
            if batch.reward[s_q_id] is None:
                r = batch.reward[s_q_id]
            else:
                r = 0
            expected_next_state_q_values.append(
                (torch.max(policy_net(torch.tensor([s_q], dtype=torch.float).to(self.Q.device))) * self.alpha) + r)
                # (torch.max(self.Q.forward(torch.tensor([s_q], dtype=torch.float).to(self.Q.device))) * self.alpha) + r)

        # turn tensor lists into a single tensor
        state_action_q_values = torch.stack(state_action_q_values)
        expected_next_state_q_values = torch.stack(expected_next_state_q_values)

        loss = self.Q.loss(state_action_q_values, expected_next_state_q_values).to(self.Q.device)
        # print(f"loss: {loss}")
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

        if EPISODE % UPDATE_PERIOD == 0:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{EPISODE}")
            self.Q.load_state_dict(policy_net.state_dict())

        # rewards = torch.tensor(batch.reward).to(self.Q.device)
        return loss
        # reward avrg, loss, position

# robot module instance
#TODO: reset the GPS per episode
class Module:
    def __init__(self, robot_, bot_id, database, agent_=None):
        #  webots section
        self.robot = robot_
        self.database = NNDataset(database)
        # self.supervisor = Supervisor()
        # self.robot_node = self.supervisor.getFromDef(self.robot.getName())
        # self.trans_field = self.robot_node.getField("translation")
        # self.trans_field = self.robot.getField("translation")
        self.timeStep = int(self.robot.getBasicTimeStep())
        self.prev_episode = EPISODE
        self.episode_reward = 0
        # self.name = name
        self.bot_id = bot_id
        self.self_message = bytes

        self.gps = self.robot.getDevice("gps")
        self.gps.resolution = 0
        self.gps.enable(self.timeStep)

        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.timeStep)

        self.motor = self.robot.getDevice("motor")

        # making all modules have NN
        self.agent = agent_
        self.receiver.setChannel(0)
        self.emitter.setChannel(0)

        self.actions = [
            Action("Up", self.action_up),
            Action("Down", self.action_down),
            Action("Neutral", self.action_neutral)
        ]
        self.action_reset = False
        self.old_pos = self.gps.getValues()
        self.new_pos = self.old_pos
        self.t = 0
        self.batch_ticks = 0
        self.reward = 0
        self.loss = None
        self.calc_reward()

        self.global_states = [1]*NUM_MODULES
        self.prev_global_states = [1]*NUM_MODULES
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
        print(f"[*] {self.bot_id} - initial action: {self.current_action}")
        self.prev_actions = self.current_action

        self.state_changer()
        self.notify_of_a_state(self.current_state)

    # sends the leader-decided values to all of the modules
    def notify_of_a_state(self, state):
        # setting system states
        message = struct.pack("i", self.bot_id)
        message += struct.pack("i", state)

        self.global_states[self.bot_id-1] = state
        # print(f"[*] {self.bot_id} - state: {state}")
        ret = self.emitter.send(message)
        if ret == 1:
            pass
            # print(f"[*] {self.bot_id} - acts were sent")
        elif ret == 0:
            print(f"[*] {self.bot_id} - acts were not sent")
            exit(3)
        else:
            print(f"[*] {self.bot_id} - error while sending data")
            exit(123)

        self.self_message = message

    # gets actions which were sent by the leader
    def get_other_module_states(self):
        if self.receiver.getQueueLength() > 0:
            while self.receiver.getQueueLength() > 0:
                message = self.receiver.getData()
                data = struct.unpack("i" + "i", message)
                self.global_states[data[0]-1] = data[1]
                self.receiver.nextPacket()
        else:
            print("Error: no packets in receiver queue.")
            return 2

    # old action break down
    def action_up(self):
        self.motor.setPosition(1.57)

    def action_down(self):
        self.motor.setPosition(-1.57)

    def set_default_action(self):
        self.motor.setPosition(0)

    def action_neutral(self):
        # TODO: REMOVE IF HAS ODD RESULTS
        # experiment to see if making neutral not do anything will be beneficial or not


        # self.motor.setPosition(0)
        pass

    def calc_reward(self):
        x = (self.old_pos[0] - self.new_pos[0])**2
        y = (self.old_pos[2] - self.new_pos[2])**2
        self.reward = (x + y)**(1/2)

        #
        # a = np.array((self.old_pos[0], self.old_pos[2]))
        # b = np.array((self.new_pos[0], self.new_pos[2]))
        # delta = b - a
        # # print(f"delta: {delta}   a: {self.old_pos}  b: {self.new_pos}")
        # if delta[0] < 0 and delta[1] >= 0:
        #     delta[0] = 0
        # elif delta[0] >= 0 and delta[1] < 0:
        #     delta[1] = 0
        # self.reward = abs(np.sum(abs(delta)))

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

    def learn(self):
        # this is for the init step after the classes got declared
        # if self.action_reset:
        #     if self.t + (self.timeStep / 1000.0) < T:
        #         self.set_default_action()
        #         return
        #     self.action_reset = False

        if self.act is None:
            ret = self.get_other_module_states()
            if ret == 2:
                return
            # print(f"[*] {self.bot_id} global_actions: {self.global_actions}")

            # action_index = self.global_actions[self.bot_id - 1]
            # print(self.current_action)
            self.act = self.actions[self.current_action]
            # print(f"{self.bot_id} {self.prev_global_states}  {self.global_states}")
            # exit(33)
            # self.current_module_state = self.global_actions

        elif self.t + (self.timeStep / 1000.0) < T:
            self.act.func()

        elif self.t < T:
            self.new_pos = self.gps.getValues()
            self.calc_reward()



            if self.reward != self.reward or self.reward is None:
                pass
            else:
                # print(self.reward)

                # print(f"[*] {self.bot_id} reward for nn: {self.reward}")

                state = self.global_states  # before taking new action
                # print(f"[*] {self.bot_id} state for nn: {state}")
                # state_ = self.current_module_state  # after taking new action
                state_ = self.prev_global_states  # after taking new action
                # print(f"[*] {self.bot_id} state_ for nn: {state_}")

                #  mean action
                states = sum(state) / NUM_MODULES
                states_ = sum(state_) / NUM_MODULES

                # actions = self.global_actions[self.bot_id-1]
                actions = self.current_action
                # rewards = self.reward

                # states = torch.tensor([states], dtype=torch.float).to(self.agent.Q.device)
                # states = torch.tensor([states], dtype=torch.float)
                states = [states]
                # actions = torch.tensor(actions).to(self.agent.Q.device)
                # actions = torch.tensor(actions)
                actions = actions
                # print(f"reward: {self.reward}")
                # print(f"{self.reward} {type(self.reward)}")

                # rewards = torch.tensor(self.reward, dtype=torch.float).to(self.agent.Q.device)
                # rewards = torch.tensor(self.reward, dtype=torch.float)
                rewards = self.reward
                # states_ = torch.tensor([states_], dtype=torch.float).to(self.agent.Q.device)
                # states_ = torch.tensor([states_], dtype=torch.float)
                states_ = [states_]

                # if self.bot_id == 1:
                #     print(f"{states}           {actions}              {states_}               {rewards}")
                # self.database.add_data("{}\n".format(Transition(states, actions, states_, rewards)))
                self.database.add_data("{},{},{},{}\n".format(states[0], actions, states_[0], rewards))
                # memory.push(states, actions, states_, rewards)
                self.episode_reward += self.reward

                if EPISODE > self.prev_episode:
                    if self.bot_id == LEADER_ID:
                        # print(f"Reward: {self.episode_reward}")
                        writer(self.bot_id, NUM_MODULES, TOTAL_ELAPSED_TIME, self.episode_reward, self.loss)
                    self.episode_reward = 0
                    self.prev_episode = EPISODE

                if self.batch_ticks > BATCH_SIZE:

                    self.loss = self.agent.learn()


                    # print(loss.item(), self.current_action, torch.mean(rewards_).item(), self.tick)
                    # TODO: add logging here

                    # if self.tick > MAX_EPISODES:
                    #     exit()
                    # else:
                    #     self.tick += 1

                    # if len(memory) < memory.capacity:
                    if self.database.__len__() < MEMORY_CAPACITY:
                        self.batch_ticks = 0
                    else:
                        print(f"**[{self.bot_id}]** end of learning period")
                        exit(33)
                else:
                    self.batch_ticks += 1

            # TODO: change to neighbor modules later on
            # new_action = self.agent.choose_action(sum(self.leader_decided_actions)/NUM_MODULES)
            self.prev_actions = self.current_action
            self.current_action = self.agent.choose_action(sum(self.global_states)/NUM_MODULES)[0]
            self.prev_global_states = self.global_states
            self.state_changer()
            self.notify_of_a_state(self.current_state)

        else:
            self.act = None
            self.old_pos = self.gps.getValues()

            self.t = 0
#
#

"""
An episode is representing a single trial


USE THE SAME LOSS FOR THE SECTION - so for every 60 logs the loss is the same and then it changes adn so on


experience is result of action taken

store reserve models for episodes 

each episode is 30 min of simulation time
"""
def writer(name, num_of_bots, time_step, reward, loss):
    # loss reward location(GPS)     name num_of_bots
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

    file_name = "{}_MODULES_{}.txt".format(num_of_bots, name)
    file_path = os.path.join(current_run, file_name)
    # if not os.path.isfile(file_path):
    #     with open(file_path, "w") as fin:
    #         fin.write("{} RUN {} EPISODES for {} MODULE SYSTEM for link {}\n".format(DATE_TODAY,
    #                                                                                    MAX_EPISODES, num_of_bots, name))
    #         fin.write("TIME_STEP,Reward,Loss,Location x,Location y,Location zExperience\n")
    with open(file_path, "a") as fin:
        fin.write('{},{},{},{}\n'.format(time_step, reward, loss, EPISODE))


def db_maker(bot_id):
    db_path = os.path.join(os.getcwd(), "DBs")
    if not os.path.isdir(db_path):
        try:
            os.mkdir(db_path)
        except FileExistsError:
            pass

    set_folder_path = os.path.join(db_path, "{}_MODULES".format(NUM_MODULES))
    if not os.path.isdir(set_folder_path):
        try:
            os.mkdir(set_folder_path)
        except FileExistsError:
            pass

    files_in_dir = os.listdir(set_folder_path)
    # format: db_NUM_MODULES_number.txt
    if files_in_dir:
        last = 0
        # only valid for first 9 databases
        for f in files_in_dir:
            if last < int(f[-5]) and int(f[5]) == bot_id:
                last = int(f[-5])
        filename = "db_{}_{}_{}.txt".format(NUM_MODULES, bot_id, last+1)
        filename = os.path.join(set_folder_path, filename)
    else:
        filename = "db_{}_{}_0.txt".format(NUM_MODULES, bot_id)
        filename = os.path.join(set_folder_path, filename)

    os.mknod(filename)
    # with open(filename, "a") as fin:
    #     fin.write("")

    return filename

#
#
# def stat_to_img():
#     # plot by 10 lines
#     file = open(str(TRIAL_PATH), "r", newline="\n")
#     reader = csv.reader(file, delimiter=',')
#     times = []
#     rewards = []
#
#     for r in reader:
#         times.append(r[0])
#         rewards.append(r[3])
#     plt.plot(rewards, times, label="Reward vs time", color='r')
#     plt.xlabel("Time")
#     plt.ylabel("Reward")
#     plt.title("Reward vs time for {}".format(TRIAL_NAME))
#     temp = TRIAL_PATH
#     # temp = temp.replace(TRIAL_NAME, "{}_img.png".format(TRIAL_NAME[:-4]))
#     temp = temp.replace(TRIAL_NAME, "")
#     temp = os.path.join(temp, "imgs")
#
#     if not os.path.isdir(temp):
#         os.mkdir(temp)
#     temp = os.path.join(temp, "{}_img.png".format(TRIAL_NAME[:-4]))
#     plt.show()
#     plt.savefig(temp, dpi=100000)
#
#     exit(1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    print(f"Starting the training: {time.strftime('%H:%M:%S', time.localtime())}")
    robot = Supervisor()
    # self.supervisor = Supervisor()
    # robot = supervisor.getFromDef("MODULE_2")
    # self.trans_field = self.robot_node.getField("translation")
    # self.trans_field = self.robot.getField("translation")
    eps_history = []
    filename = db_maker(int(robot.getName()[7]))

    # every module has NN
    module = Module(robot, int(robot.getName()[7]), filename, agent_=Agent(int(robot.getName()[7]), NUM_MODULES,  3, 0.001,
                                                                           filename,
                                                                           alpha=ALPHA, epsilon=EPSILON,
                                                                           eps_dec=0.001, eps_min=MIN_EPSILON))
    assign_ = False
    learn = True

    i = 0
    while i < 100:
        i += 1
        time.sleep(0.05)
    print(f"Finished buffer period in: {time.time()-start_time} ===== {time.strftime('%H:%M:%S', time.localtime())}")
    last_episode = time.time()
    while module.robot.step(module.timeStep) != -1:
        i = 0
        while i < 1000:
            i += 1

        module.learn()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # exit()
        module.t += module.timeStep / 1000.0
        TOTAL_ELAPSED_TIME += module.timeStep / 1000.0
        if 0 <= TOTAL_ELAPSED_TIME % 60 <= 1:
            if not assign_:
                EPISODE += 1
                if module.bot_id == LEADER_ID:
                    print(f"Episode: {EPISODE} -- "
                          f"{time.time() - start_time} ===== time since last episode: {time.time() - last_episode}")
                assign_ = True
            module.motor.setPosition(0)
            # print("REST")
        else:
            assign_ = False
        #     print(f"{robot.getName().upper()}")
        #     trans_field = robot.getRoot()
        #     print(f"{robot.get}")
        #     trans_field = trans_field.getField("translation")
        #     trans_field.setSFVec3f(INITIAL)
            # self.robot.resetPhysics()
        # if TOTAL_ELAPSED_TIME > 3600.0:
        # SUMMING THE REWARD
        if EPISODE > MAX_EPISODE:
            end = time.time()
        # if TOTAL_ELAPSED_TIME > 600.0:
            # if TOTAL_ELAPSED_TIME > 36.0:
            print(f"Runtime: {end - start_time}")
            print("LOGGING OFF")
            exit()
