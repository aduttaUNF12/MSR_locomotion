import math
from datetime import date

from controller import Robot, Supervisor, Node, Field
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


__version__ = '06.21.21'


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
replay_buf_mask = ReplayBuffer(shape=(MEMORY_CAPACITY,))


class Action:
    def __init__(self, name, function):
        self.name = name
        self.func = function


# the NN itself
class LinearDQN(nn.Module):
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
        # self.database = NNDataset(database)
        self.updated = False

        self.action_space = [i for i in range(self.n_actions)]

        # q estimate
        self.Q = LinearDQN(self.number_of_modules, self.lr, n_actions)
        self.Q.load_state_dict(policy_net.state_dict())
        self.Q.eval()

    # Greedy action selection
    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):

            state = torch.tensor([module_actions], dtype=torch.float).to(self.Q.device)
            action = self.Q.forward(state)
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
        self.Q.optimizer.zero_grad()

        # TODO: priority memory
        random_sample = np.random.randint(0, replay_buf_state.head, BATCH_SIZE, np.int)
        sample = replay_buf_state_.get(random_sample)
        states__t = torch.tensor(sample, dtype=torch.float).to(self.Q.device)

        state_action_q_values = []
        batch_sates = replay_buf_state.get(random_sample)
        batch_action = replay_buf_action.get(random_sample)
        batch_reward = replay_buf_reward.get(random_sample)
        for s_t, a in zip(batch_sates, batch_action):

            state_action_q_values.append(torch.gather(
                self.Q.forward(torch.tensor([s_t], dtype=torch.float).to(self.Q.device)),
                0, torch.tensor(a, dtype=torch.int64).to(self.Q.device)))

        expected_next_state_q_values = []
        #  get q value for state_
        for s_q_id, s_q in enumerate(states__t):
            if batch_reward[s_q_id] is None:
                r = batch_reward[s_q_id]
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

        if EPISODE % UPDATE_PERIOD == 0 and self.updated is False:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{EPISODE}")
            self.Q.load_state_dict(policy_net.state_dict())
            # global BASE_LOGS_FOLDER
            if BASE_LOGS_FOLDER is not None:
                torch.save(module.agent.Q.state_dict(), os.path.join(BASE_LOGS_FOLDER, "agent.pt"))

            self.updated = True
        elif EPISODE % UPDATE_PERIOD != 0:
            self.updated = False
        # rewards = torch.tensor(batch.reward).to(self.Q.device)
        return loss

# robot module instance
#TODO: reset the GPS per episode
# class Module(Supervisor):
class Module(Supervisor):
    # def __init__(self, robot_, bot_id, database, agent_=None):
    def __init__(self):
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
        self.agent = Agent(self.bot_id, NUM_MODULES, 3, 0.001, "null", alpha=ALPHA,
                           epsilon=EPSILON, eps_dec=0.001, eps_min=MIN_EPSILON)
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
        self.notify_of_a_state(self.current_action)

    # notify other modules of the chosen action
    # TODO: fix phrasing form ___of_a_state to ___of_an_action, it's confusing when taking mean and so on
    def notify_of_a_state(self, state):
        # setting system states
        message = struct.pack("i", self.bot_id)
        message += struct.pack("i", state)

        self.global_states[self.bot_id-1] = state
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

    # no action is taken, module just passes
    def action_neutral(self):
        # TODO: REMOVE IF HAS ODD RESULTS
        # experiment to see if making neutral not do anything will be beneficial or not


        # self.motor.setPosition(0)
        pass

    # calculates reward based on the travel distance from old position to the new, calculates hypotenuse of two points
    def calc_reward(self):
        self.reward = math.hypot(self.old_pos[0]-self.new_pos[0], self.old_pos[2]-self.new_pos[2])

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

    def learn(self):
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

                replay_buf_reward.put(np.array(self.reward))
                replay_buf_state.put(np.array(self.current_state))
                replay_buf_state_.put(np.array(self.prev_state))
                try:
                    replay_buf_action.put(np.array(sum(self.global_states)/NUM_MODULES))
                except TypeError:
                    print(f"global_states: {self.global_states}  ({type(self.global_states)})")
                    print(f"sum(gs): {sum(self.global_states)} ({type(sum(self.global_states))})")
                    s = sum(self.global_states)
                    replay_buf_action.put(np.array(s.cpu()))
                    # print(f"np.array(s(sg)): {np.array(sum(self.global_states)/NUM_MODULES)} ({type(np.array(sum(self.global_states)/NUM_MODULES))})")
                    # exit(22)
                # for MFRL memory
                #     state will keep track of this
                replay_buf_state.return_buffer_len = \
                    min(MEMORY_CAPACITY, replay_buf_state.return_buffer_len + self.batch_ticks)

                # memory.push(states, actions, states_, rewards)
                self.episode_reward += self.reward

                if EPISODE > self.prev_episode:
                    # self.simulationReset()
                    if self.bot_id == LEADER_ID:
                        print(f"Buffer len: {replay_buf_state.return_buffer_len} "
                              f"{(replay_buf_state.return_buffer_len/MEMORY_CAPACITY)*100}%"
                              f" Loss: {self.loss}")
                        # print(f"Reward: {self.episode_reward}")
                        writer(self.bot_id, NUM_MODULES, TOTAL_ELAPSED_TIME, self.episode_reward, self.loss)
                    self.episode_reward = 0
                    self.prev_episode = EPISODE

                if self.batch_ticks > BATCH_SIZE:

                    self.loss = self.agent.learn()

                    self.batch_ticks = 0

                else:
                    self.batch_ticks += 1

            # TODO: change to neighbor modules later on
            self.prev_actions = self.current_action
            self.current_action = self.agent.choose_action(sum(self.global_states)/NUM_MODULES)[0]
            self.state_changer()
            self.notify_of_a_state(self.current_action)

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


# makes a file for the replay memory and stores replays there as to not overload the RAM
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

    return filename


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
                          f"{time.time() - start_time} ===== time since last episode: {time.time() - last_episode}")
                    last_episode = time.time()

                assign_ = True
                module.simulationReset()
                self.old_pos = self.gps.getValues() # AD's change -- position resets to the initial position. 
            #module.motor.setPosition(0)
        else:
            assign_ = False
        if EPISODE > MAX_EPISODE:
            end = time.time()
            print(f"Runtime: {end - start_time}")
            print("LOGGING OFF")
            exit()
