import math
import random
import struct
import sys
import os
import itertools
from collections import namedtuple
from datetime import date

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from controller import Supervisor


__version__ = '08.28.21'


# Constants
EPSILON = 1
GAMMA = 0.1
MIN_EPSILON = 0.1
T = 0.1
# BATCH_SIZE = 128
# BATCH_SIZE = 64
# BATCH_SIZE = 5
# BATCH_SIZE = 2
# BATCH_SIZE = 32
BATCH_SIZE = 10
MAX_EPISODE = 5000
# EPISODE_LIMIT = 200   # Limiting Replay Memory to 200 Episodes
# EPISODE_LIMIT = 50   # Limiting Replay Memory to 200 Episodes
EPISODE_LIMIT = 12   # Limiting Replay Memory to 200 Episodes
# EPISODE_LIMIT = 5   # Limiting Replay Memory to 200 Episodes
# MEMORY_CAPACITY = 10**10
# MEMORY_CAPACITY = 2**30
# MEMORY_CAPACITY = 2**22  # this equals to a memory pool of over 4 mil spots, and with 5000 episodes each having 577
#                       of data we would only need just over 3 mil spots, so this should work and optimize things a bit
MEMORY_CAPACITY = 10**6  # 1M in memory allocation
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
NN_TYPE = None
POLICY_NET = None
TARGET_NET = None

REPLAY_MEMORY_EPISODE = 1
ReplayMemory_EpisodeBuffer = {}


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
replay_buf_mean_action = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
#  TODO: possibly rework these \/
Replay_Buf_Vector_States = []
Replay_Buf_Vector_States_ = []
Replay_Buf_Vector_Mean_Actions = []
Replay_Buf_Vector_Mean_Actions_ = []

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

        self.conv1 = nn.Conv2d(in_channels=3*(number_of_modules + 1), out_channels=32, kernel_size=(1, 1)).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)).to(self.device)
        self.bn2 = nn.BatchNorm2d(64).to(self.device)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)).to(self.device)
        self.bn3 = nn.BatchNorm2d(128).to(self.device)

        x = torch.rand((32, 3*(number_of_modules + 1))).to(self.device).view(32, 3*(number_of_modules + 1), 1, 1)

        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 515).to(self.device)
        self.fc2 = nn.Linear(515, 3).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def convs(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, actions):
        x = self.convs(actions)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x[0]


class FCNN(nn.Module):
    def __init__(self, number_of_modules, lr, n_actions):
        super(FCNN, self).__init__()
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        self.fc1 = nn.Linear(3*(number_of_modules + 1), 32).to(self.device)
        self.fc2 = nn.Linear(32, 64).to(self.device)
        self.fc3 = nn.Linear(64, self.n_actions).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, actions):
        x = F.relu(self.fc1(actions))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x[0]


# policy_net = CNN(NUM_MODULES, 0.001, 3)
# agent who controls NN, in this case the main module (module 1)


class Agent:
    # policy_net = does all of the training and tests
    # target_net = final network

    def __init__(self, module_number, number_of_modules, n_actions, lr, gamma=0.99,
                 epsilon=1, eps_dec=1e-5, eps_min=0.01):
        self.module_number = module_number   # only used for logging
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.updated = False

        self.action_space = [i for i in range(self.n_actions)]

        # q estimate
        # self.target_net = CNN(self.number_of_modules, self.lr, n_actions)
        self.policy_net = POLICY_NET
        # self.target_net = NET_TYPE(self.number_of_modules, self.lr, n_actions)
        self.target_net = TARGET_NET
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    @staticmethod
    def payload_maker(temp_):
        if NN_TYPE == "FCNN":
            payload = torch.tensor([temp_], dtype=torch.float).to(POLICY_NET.device)
        else:
            payload = torch.tensor([temp_]*32, dtype=torch.float).to(POLICY_NET.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
        return payload

    # Greedy action selection
    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):
            # make list of lists into a single list, and turn it into numpy array
            temp_ = np.array(list(itertools.chain(*module_actions)))
            # [temp_]*32 is needed to fit into the Conv2d (just makes a list of lists of temp_)
            # if NN_TYPE == "FCNN":
            #     payload = torch.tensor([temp_], dtype=torch.float).to(POLICY_NET.device)
            # else:
            #     payload = torch.tensor([temp_]*32, dtype=torch.float).to(POLICY_NET.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
            payload = self.payload_maker(temp_)
            action = self.policy_net.forward(payload)
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

    def optimize(self, batch=False):
        self.policy_net.optimizer.zero_grad()

        # get the current amount of Episodes
        episodes = range(len(ReplayMemory_EpisodeBuffer))[1:]
        # Not necessary, but was used to testing running optimizer after every Episode as well as a batch (NOT USED)
        if len(ReplayMemory_EpisodeBuffer) >= BATCH_SIZE and batch is True:
            sample = np.random.choice(episodes, BATCH_SIZE-1, replace=False)
        else:
            sample = np.array([len(ReplayMemory_EpisodeBuffer)])
        del episodes

        ranges = []

        for s in sample:
            # This is done mainly because all of the Episodes contain 577 actions, but some have 576
            # (it might just be the first Episode that only has 576 but I need to look further into it,
            # for now this works)
            if ReplayMemory_EpisodeBuffer[s]['max'] - ReplayMemory_EpisodeBuffer[s]['min'] > 576:
                sub = ReplayMemory_EpisodeBuffer[s]['max'] - ReplayMemory_EpisodeBuffer[s]['min'] - 576
                ReplayMemory_EpisodeBuffer[s]['max'] = ReplayMemory_EpisodeBuffer[s]['max'] - sub
            # make a list of numbers from min to max and add that list to ranges list (each list contains
            #   actions taken during the Episode
            ranges.append(np.arange(ReplayMemory_EpisodeBuffer[s]['min'],
                                    ReplayMemory_EpisodeBuffer[s]['max']))

        del sample

        state_action_values = []
        expected_state_action_values = []
        # iterates over the array of range arrays,  index1 is the id of the range array, r is the array or ranges

        if self.module_number == LEADER_ID:
            print(f"Inside of the optimizer\nvector sizes:\nstates: {len(Replay_Buf_Vector_States)}\n"
                  f"States_: {len(Replay_Buf_Vector_States_)}\n"
                  f"Mean_Actions: {len(Replay_Buf_Vector_Mean_Actions)}\n"
                  f"Mean_Actions_: {len(Replay_Buf_Vector_Mean_Actions_)}\n")

        for index1, r in enumerate(ranges):
            # pull values corresponding to the r range
            temp_action = replay_buf_action.get(r)
            temp_rewards = replay_buf_reward.get(r)

            # iterates over the r, index2 is the id of the item in r, and item is the entry in the list
            for index2, item in enumerate(r):
                # get the list of state vectors
                robot_state_vectors = Replay_Buf_Vector_States[item][0:NUM_MODULES]
                # add corresponding mean action
                robot_state_vectors.append(Replay_Buf_Vector_Mean_Actions[item])
                # convert a list of lists into a single list
                temp_ = np.array(list(itertools.chain(*robot_state_vectors)))
                # if self.module_number == LEADER_ID:
                #     print(f"temp_: {temp_}")
                # if NN_TYPE == "FCNN":
                #     payload = torch.tensor([temp_], dtype=torch.float).to(POLICY_NET.device)
                # else:
                #     payload = torch.tensor([temp_]*32, dtype=torch.float).to(POLICY_NET.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
                # payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
                payload = self.payload_maker(temp_)
                # if self.module_number == LEADER_ID:
                #     print(f"payload: {payload}")
                res = self.policy_net(payload.to(self.policy_net.device))
                # if self.module_number == LEADER_ID:
                #     print(f"res: {res}")
                # exit(111)
                res = res.to('cpu')
                # res[temp_action[index2]] = select the estimate value form the res list which corresponds to an
                #   action of at the same index; adds the value in tensor form to the state_action_values
                state_action_values.append(torch.tensor(res[temp_action[index2]], dtype=torch.float, requires_grad=True).to(self.policy_net.device))

                # try:
                #     state_action_values.append(torch.tensor(res[temp_action[index2]], dtype=torch.float, requires_grad=True).to(self.policy_net.device))
                # except IndexError:
                #     if self.module_number == LEADER_ID:
                #         print(f"Got index Error\nranges: {ranges}\nr: {r}\nres: {res}\ntemp_action {temp_action}\nindex 2 {index2}")
                #     exit(111)
                del res

            for index3, item in enumerate(r):
                # get the list of state vectors
                robot_state_vectors_ = Replay_Buf_Vector_States_[item][0:NUM_MODULES]
                # add corresponding mean action
                robot_state_vectors_.append(Replay_Buf_Vector_Mean_Actions_[item])
                # convert a list of lists into a single list
                temp_ = np.array(list(itertools.chain(*robot_state_vectors_)))

                # payload = torch.tensor([temp_]*32, dtype=torch.float, requires_grad=False).to(self.target_net.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
                payload = self.payload_maker(temp_)
                res = self.target_net(payload)
                # get the position of the largest estimate in the res list, multiplies it by gamma and
                #   adds reward associated with this action in a given Episode
                expected_state_action_values.append((torch.tensor(np.argmax(res.to('cpu').detach().numpy()), dtype=torch.float).to(self.target_net.device) * self.gamma) + temp_rewards[index3])
                del res
            del item
        # print("loops are over")
        # torch.stack([tensor]) combines all tensors in a list into a single tensor
        state_action_values = torch.stack(state_action_values)
        state_action_values = state_action_values.double().float()

        expected_state_action_values = torch.stack(expected_state_action_values).double().float()

        loss = self.policy_net.loss(state_action_values, expected_state_action_values)
        loss.backward()
        self.policy_net.optimizer.step()
        self.decrement_epsilon()

        if EPISODE % UPDATE_PERIOD == 0 and self.updated is False:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{EPISODE}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

            if BASE_LOGS_FOLDER is not None:
                torch.save(module.agent.target_net.state_dict(), os.path.join(BASE_LOGS_FOLDER, "agent.pt"))

            self.updated = True
        elif EPISODE % UPDATE_PERIOD != 0:
            self.updated = False

        return loss


# robot module instance
class Module(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        #  webots section
        self.bot_id = int(self.getName()[7])
        self.timeStep = int(self.getBasicTimeStep())
        self.prev_episode = 1
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_mean_action = []
        self.prev_episode_mean_action = []
        self.episode_current_action = []

        self.self_message = bytes

        self.gps = self.getDevice("gps")
        self.gps.resolution = 0
        self.gps.enable(self.timeStep)

        self.emitter = self.getDevice("emitter")

        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.timeStep)

        self.motor = self.getDevice("motor")

        # making all modules have NN
        self.agent = Agent(self.bot_id, NUM_MODULES, 3, 0.001, gamma=GAMMA,
                           epsilon=EPSILON, eps_dec=0.001, eps_min=MIN_EPSILON)

        self.receiver.setChannel(5)

        self.emitter.setChannel(5)

        self.actions = [
            Action("Up", self.action_up),
            Action("Down", self.action_down),
            Action("Neutral", self.action_neutral)
        ]

        self.old_pos = self.gps.getValues()
        self.new_pos = self.old_pos
        self.t = 0
        self.batch_ticks = 1
        self.reward = 0
        self.loss = None
        self.calc_reward()

        self.global_actions = [1]*NUM_MODULES
        self.global_states = [1]*NUM_MODULES
        self.global_states_vectors = []
        self.global_actions_vectors = []
        #  TODO: change mean action to a vector representation
        self.mean_action_vector = []
        self.mean_action = 0
        self.prev_states_vector = []
        self.prev_mean_action_vector = []
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
        self.sts = None


        # getting leader to decide initial actions
        self.current_action = np.random.choice([i for i in range(3)], 1)[0]
        self.prev_actions = self.current_action

        # TEMP FOR ERROR CHECKING TODO
        self.tries = 0
        self.batch_updated = False

        self.min_max_set = False
        self.min_batch = 0
        self.max_batch = 0
        self.recycles = 0

        self.state_changer()
        self.notify_of_an_action(self.current_action)
        # self.notify_of_an_action(self.current_state, action=False)

    # notify other modules of the chosen action
    # if action is True means that an action is being transmitted, if False current state is being transmitted
    def notify_of_an_action(self, state, action=True):
        message = struct.pack("i", self.bot_id)
        if action:
            # 1 if sending action
            self.global_actions[self.bot_id-1] = state
            message += struct.pack("i", 0)
        else:
            # 2 if sending state
            self.global_states[self.bot_id-1] = state
            message += struct.pack("i", 1)

        message += struct.pack("i", state)

        ret = self.emitter.send(message)

        if ret == 1:
            pass
        elif ret == 0:
            print(f"[*] {self.bot_id} - acts were not sent")
            exit(3)
        else:
            print(f"[*] {self.bot_id} - error while sending data")
            exit(123)

        # self.self_message = message

    # gets actions which were sent by the leader
    # if action is True means that an action is being received, if False current state is being received
    def get_other_module_actions(self):
        if self.receiver.getQueueLength() > 0:
            while self.receiver.getQueueLength() > 0:
                message = self.receiver.getData()
                # data = [bot id, action/state flag, payload]
                data = struct.unpack("i" + "i" + "i", message)
                # print(f"[{self.bot_id}] got emitter packet: {data}", file=sys.stderr)

                if data[1] == 0:
                    self.global_actions[data[0]-1] = data[2]
                else:
                    self.global_states[data[0]-1] = data[2]

                self.receiver.nextPacket()
        else:
            # print(f"[{self.bot_id}] Error: no packets in receiver queue.")
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

    # Converts global_actions (actions of all of the modules in the chain) into a vector representation
    def actions_to_vectors(self):
        # adds vector representation of the state to an array, [x, y, z]
        for action in self.global_actions:
            if int(action) == 1:
                self.global_actions_vectors.append([0, 0, 0])
            elif int(action) == 0:
                self.global_actions_vectors.append([0, -1, 0])
            elif int(action) == 2:
                self.global_actions_vectors.append([0, 1, 0])
            else:
                print(f"[{self.bot_id}] Error with state_to_vectors !!!!!!!!!!!!!", file=sys.stderr)
                exit(2)

        self.calc_mean_action_vector()
        self.global_actions_vectors = []

    # Converts global states to vector form
    def states_to_vectors(self):
        """
       states:  0 - min
                1 - zero
                2 - max
       """
        # adds vector representation of the state to an array, [Down, Neutral, Up]
        # reasoning for line 517 is calc_mean_action_vector()
        Replay_Buf_Vector_States_.append(self.prev_states_vector[0:NUM_MODULES] if len(self.prev_states_vector) >= NUM_MODULES else [[0, 0, 0]]*NUM_MODULES)

        self.global_states_vectors = []
        for state in self.global_states:
            if int(state) == 1:
                self.global_states_vectors.append([0, 0, 0])
            elif int(state) == 0:
                self.global_states_vectors.append([1, 0, 0])
            elif int(state) == 2:
                self.global_states_vectors.append([0, 0, 1])
            else:
                print(f"[{self.bot_id}] Error with state_to_vectors !!!!!!!!!!!!!", file=sys.stderr)
                exit(2)

        # for some reason there is an extra [0] array at the end
        Replay_Buf_Vector_States.append(self.global_states_vectors[0:(NUM_MODULES)])

    # Loops through NUM_MODULES+1 (since there is no module 0), sums actions of all modules except for current
    # divides by the total number of modules
    def calc_mean_action(self):
        a = 0
        for m in range(NUM_MODULES):
            # self.bot_id - 1 since bot ids start with 1 while arrays with index of 0
            if m != (self.bot_id - 1):
                try:
                    a += self.global_actions[m]
                except IndexError:
                    pass
        # NUM_MODULE - 1 since we are taking a mean of all except current modules
        self.prev_mean_action = self.mean_action
        self.mean_action = a / (NUM_MODULES - 1)
        # logger(mean_action=self.mean_action, a=a)
        # self.mean_action_rounder()

    # Calculates the mean of the action vectors
    def calc_mean_action_vector(self):
        # Making a base vector the shape of (1, NUM_MODULES)
        a = [0]*NUM_MODULES
        # add previous mean action vector to the buffer
        #   if previous mean action vector has size greater than/equal to number of modules, this means that either
        #   previous mean action vector has more values than actions so limit the input to only the first NUM_MODULES
        #   actions; else the previous mean action vector is just being initialized so fill with 0 vectors
        Replay_Buf_Vector_Mean_Actions_.append(self.prev_mean_action_vector[0:NUM_MODULES] if len(self.prev_mean_action_vector) >= NUM_MODULES else [0]*NUM_MODULES)
        for m in range(NUM_MODULES):
            # self.bot_id - 1 since bot ids start with 1 while arrays with index of 0
            if m != (self.bot_id - 1):
                try:
                    a = np.add(a, self.global_actions_vectors[m])
                except IndexError:
                    pass
        self.prev_mean_action_vector = self.mean_action_vector
        # NUM_MODULE - 1 since we are taking a mean of all except current modules
        self.mean_action_vector = np.true_divide(a, (NUM_MODULES - 1))
        Replay_Buf_Vector_Mean_Actions.append(self.mean_action_vector[0:NUM_MODULES])

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
            # print(f"[{self.bot_id}] got all of the actions: {self.global_actions}", file=sys.stderr)
            self.tries = 0
            # select an action corresponding to the current_action number
            self.act = self.actions[self.current_action]
            # self.calc_mean_action()
            self.actions_to_vectors()
            self.calc_mean_action_vector()

            self.notify_of_an_action(self.current_state, action=False)

        if self.sts is None:
            # print(f"[{self.bot_id}] inside self.sts None", file=sys.stderr)

            ret = self.get_other_module_actions()
            if ret == 2:
                if self.tries > 100:
                    print("Error with tries")
                    exit(11)
                self.tries += 1
                return

            # print(f"[{self.bot_id}] got all of the states: {self.global_states}", file=sys.stderr)
            self.tries = 0
            self.sts = 1
            self.states_to_vectors()

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
                if not self.min_max_set:
                    self.min_batch = replay_buf_state.return_buffer_len
                    self.min_max_set = True

                replay_buf_state.return_buffer_len += 1
                if replay_buf_state.return_buffer_len > MEMORY_CAPACITY:
                    print(f"Replay buffer is full")
                    exit(11)

                self.episode_reward += self.reward
                self.episode_rewards.append(self.reward)
                self.episode_mean_action.append(self.mean_action)
                self.prev_episode_mean_action.append(self.prev_mean_action)

                self.episode_current_action.append(self.current_action)

                # If Episode changed
                if EPISODE > self.prev_episode:
                    global REPLAY_MEMORY_EPISODE

                    self.max_batch = replay_buf_state.return_buffer_len
                    self.min_max_set = False
                    ReplayMemory_EpisodeBuffer[EPISODE-1] = {"min": self.min_batch,
                                                             "max": self.max_batch}

                    # Since Episode 1 usually is 1 less than all other episodes
                    if EPISODE - 1 == 1:
                        ReplayMemory_EpisodeBuffer[REPLAY_MEMORY_EPISODE] = {"min": self.min_batch,
                                                                             "max": self.max_batch + 1}
                    else:
                        ReplayMemory_EpisodeBuffer[REPLAY_MEMORY_EPISODE] = {"min": self.min_batch,
                                                                             "max": self.max_batch}

                    replay_buf_reward.put(np.array(self.episode_rewards))
                    replay_buf_state.put(np.array(self.current_state))
                    replay_buf_mean_action.put(np.array(self.episode_mean_action))
                    replay_buf_state_.put(np.array(self.prev_state))
                    replay_buf_action.put(np.array(self.episode_current_action))

#                     if REPLAY_MEMORY_EPISODE == EPISODE_LIMIT:
#                         replay_buf_action.clear()
#                         replay_buf_state_.clear()
#                         replay_buf_reward.clear()
#                         replay_buf_state.clear()
#                         replay_buf_mean_action.clear()
#                         REPLAY_MEMORY_EPISODE = 0
#                         self.min_batch = 0
#                         self.max_batch = 0
#                         replay_buf_state.return_buffer_len = 0
#                         self.recycles += 1
#
#                         if self.bot_id == LEADER_ID:
#                             """
#                             Replay_Buf_Vector_States = []
# Replay_Buf_Vector_States_ = []
# Replay_Buf_Vector_Mean_Actions = []
# Replay_Buf_Vector_Mean_Actions_ = []
#                             """
#                             print(f"after the limit\nrecycles: {self.recycles}\nvector sizes:\nstates: {len(Replay_Buf_Vector_States)}\n"
#                                   f"States_: {len(Replay_Buf_Vector_States_)}\n"
#                                   f"Mean_Actions: {len(Replay_Buf_Vector_Mean_Actions)}\n"
#                                   f"Mean_Actions_: {len(Replay_Buf_Vector_Mean_Actions_)}\n")

                    if self.bot_id == LEADER_ID:
                        # logger
                        writer(self.bot_id, NUM_MODULES, TOTAL_ELAPSED_TIME, self.episode_reward, self.loss)


                    self.episode_reward = 0
                    self.episode_mean_action.clear()
                    self.prev_episode_mean_action.clear()
                    self.episode_current_action.clear()
                    self.prev_episode = EPISODE
                    self.batch_ticks += 1
                    self.batch_updated = False
                    REPLAY_MEMORY_EPISODE += 1

                # batch is at least at the minimal working size
                if self.batch_ticks >= BATCH_SIZE and not self.batch_updated:
                    # run the NN and collect loss
                    self.loss = self.agent.optimize(batch=True)
                    # if self.bot_id == LEADER_ID:
                    #     print(f"loss: {self.loss}")
                    self.batch_updated = True
                    # self.batch_ticks = 0

            # set previous action option to the new one
            self.prev_actions = self.current_action
            # get the array of state vectors
            robot_state_vectors = self.global_states_vectors[0:NUM_MODULES]
            robot_state_vectors.append(self.mean_action_vector)

            self.current_action = self.agent.choose_action(robot_state_vectors)[0]
            self.prev_states_vector = self.global_states_vectors
            self.global_states_vectors = []
            # run the action and change the state
            self.state_changer()
            # notify others of current action
            self.notify_of_an_action(self.current_action)

        else:
            self.act = None
            self.sts = None
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


# TODO: add args and kwargs to main
if __name__ == '__main__':
    # TODO: for now the NN selector is a string variable
    NN_TYPE = "FCNN"
    # NN_TYPE = "CNN"

    if NN_TYPE == "FCNN":
        TARGET_NET = FCNN(NUM_MODULES, 0.001, 3)
        POLICY_NET = FCNN(NUM_MODULES, 0.001, 3)
        # print(f"At fcc: {TARGET_NET}   {POLICY_NET}")
    elif NN_TYPE == "CNN":
        TARGET_NET = CNN(NUM_MODULES, 0.001, 3)
        POLICY_NET = CNN(NUM_MODULES, 0.001, 3)
    else:
        print("Enter the type of network to be used!!", file=sys.stderr)
        exit(1)

    import time
    start_time = time.time()
    print(f"Starting the training ({NN_TYPE}) : {time.strftime('%H:%M:%S', time.localtime())}")
    eps_history = []
    filename = "null"
    module = Module()
    assign_ = False
    learn = True
    i = 0
    episode_lens = []
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
        # if 0 <= TOTAL_ELAPSED_TIME % 30 <= 1:
        # if 0 <= TOTAL_ELAPSED_TIME % 5 <= 1:
            if not assign_:
                EPISODE += 1

                if module.bot_id == LEADER_ID:
                    temp = time.time() - last_episode
                    print(f"Episode: {EPISODE} -- "
                          f"{time.time() - start_time} ===== time since last episode: {temp} ====== Episode reward: {module.episode_reward} == Loss: {module.loss}")
                    episode_lens.append(temp)

                assign_ = True
                module.simulationReset()
                module.old_pos = module.gps.getValues()
                last_episode = time.time()
        else:
            # assign_ is a temp needed to prevent infinite loop on the first Episode
            assign_ = False
        if EPISODE > MAX_EPISODE:
            end = time.time()
            if module.bot_id == LEADER_ID:
                print(f"Ending training of {NN_TYPE}.")
                print(f"Avg time per episode: {float(sum(episode_lens)/len(episode_lens))} sec")
                temp = end - start_time
                print(f"Runtime: {temp} secs")
                print(f"Runtime: {temp/60} mins")
                print(f"Runtime: {temp/60/60} hours")
                print("LOGGING OFF")
            exit()
