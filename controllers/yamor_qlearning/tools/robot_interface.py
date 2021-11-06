import struct
import math
import sys
import os
import itertools
import collections
from collections import deque

import numpy as np
import torch

from .constants import EPSILON, NUM_MODULES, GAMMA, MIN_EPSILON, \
    T, BATCH_SIZE, LEADER_ID, BUFFER_LIMIT, NN_TYPE, COMMUNICATION
from .agent import Agent
from .buffers import ReplayBuffer
from .loggers import writer, path_maker

from controller import Supervisor


class Action:
    def __init__(self, name, function):
        self.name = name
        self.func = function


# robot module instance
class Module(Supervisor):
    def __init__(self, target_net, policy_net):
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
        # self.module = module
        self.episode = 1
        self.total_time_elapsed = 0
        self.self_message = bytes

        self.gps = self.getDevice("gps")
        self.gps.resolution = 0
        self.gps.enable(self.timeStep)

        self.emitter = self.getDevice("emitter")

        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.timeStep)

        self.motor = self.getDevice("motor")

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
        self.global_mean_action_vectors = [1]*NUM_MODULES
        self.global_states_vectors = [[0, 0, 0]]*NUM_MODULES
        self.global_states__vectors = [[0, 0, 0]]*NUM_MODULES
        self.global_actions_vectors = [[0, 0, 0]]*NUM_MODULES
        self.global_prev_actions_vectors = [[0, 0, 0]]*NUM_MODULES
        self.mean_action_vector = [0]*3
        self.mean_action = 0
        self.prev_states_vector = [0]*3
        self.prev_mean_action_vector = [0]*3
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
        self.re_adjust = False

        self.min_max_set = False
        self.min_batch = 0
        self.max_batch = 0
        self.recycles = 0

        self.best_reward = 0

        # BUFFERS
        self.REPLAY_MEMORY_EPISODE = 1
        self.ReplayMemory_EpisodeBuffer = {}

        self.episode_reward_temp = []
        self.episode_actions_temp = []
        self.episode_mean_actions_temp = []
        self.episode_states_temp = []
        self.episode_prev_states_temp = []
        self.episode_single_action_temp = []

        self.replay_buf_reward = deque(maxlen=BUFFER_LIMIT)
        self.replay_buf_action = deque(maxlen=BUFFER_LIMIT)
        self.replay_buf_single_action = deque(maxlen=BUFFER_LIMIT)
        self.replay_buf_state = deque(maxlen=BUFFER_LIMIT)
        self.replay_buf_state_ = deque(maxlen=BUFFER_LIMIT)
        self.replay_buf_mean_action = deque(maxlen=BUFFER_LIMIT)

        # making all modules have NN
        self.agent = Agent(self.bot_id, NUM_MODULES, 3, 0.001, gamma=GAMMA,
                           epsilon=EPSILON, eps_dec=0.001, eps_min=MIN_EPSILON,
                           target_net=target_net, policy_net=policy_net, nn_type=NN_TYPE)

        self.buffer_overflow = 0
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
        self.global_actions_vectors = []
        # adds vector representation of the state to an array, [x, y, z]
        for action in self.global_actions:
            if int(action) == 0:
                self.global_actions_vectors.append([0, 1, 0])
            elif int(action) == 1:
                self.global_actions_vectors.append([0, -1, 0])
            elif int(action) == 2:
                self.global_actions_vectors.append([0, 0, 0])
            else:
                print(f"[{self.bot_id}] Error with state_to_vectors !!!!!!!!!!!!!", file=sys.stderr)
                exit(2)

        self.global_actions_vectors = self.global_actions_vectors[0:NUM_MODULES]

        # self.global_prev_actions_vectors = self.global_actions_vectors
        self.calc_mean_action_vector()
        # self.global_actions_vectors = []

    # Converts global states to vector form
    def states_to_vectors(self):
        """
       states:  0 - min
                1 - zero
                2 - max
       """
        self.global_states__vectors = []
        self.global_states__vectors.extend(self.global_states_vectors)

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

        self.global_states_vectors = self.global_states_vectors[0:NUM_MODULES]

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
        # self.mean_action_rounder()

    # Calculates the mean of the action vectors
    def calc_mean_action_vector(self):
        # Making a base vector the shape of (1, NUM_MODULES)
        a = [0]*NUM_MODULES

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
        self.mean_action_vector = self.mean_action_vector[0:NUM_MODULES]

        all_vectors = []
        # mean action for all modules
        # m represents current module number
        for m in range(NUM_MODULES+1)[1:]:
            a = [0]*NUM_MODULES
            # mi represents module to be added
            for mi in range(NUM_MODULES):
                # if m does not equal current module number - 1; -1 because module
                #  numbers start with 1 and list indexing with 0, so we need to shift
                #  by one to get accurate data
                if mi != (m - 1):
                    try:
                        a = np.add(a, self.global_actions_vectors[mi])
                    except IndexError:
                        pass
            all_vectors.append(np.true_divide(a, (NUM_MODULES - 1)))

        self.global_mean_action_vectors = all_vectors[0:NUM_MODULES]

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
                # episode reward for the robot
                self.episode_reward_temp.extend([self.reward])
                # episode action for the robot
                self.episode_single_action_temp.extend([self.current_action])
                # episode action vectors for all robots
                self.episode_actions_temp.extend([self.global_actions_vectors])
                # episode mean action vectors for all robots
                self.episode_mean_actions_temp.extend([self.global_mean_action_vectors])
                # episode state vectors for all robots
                self.episode_states_temp.extend([self.global_states_vectors])


                # if self.episode == 3:
                #     if self.bot_id == 1:
                #         print(self.global_states_vectors)
                #         print(self.global_actions_vectors)
                #         print(self.global_mean_action_vectors)
                #         print(self.reward)
                #     exit(321)

                # episode previous state vectors for all robots

                # if self.bot_id == 1:
                #     print(f"global_states__vectors in training >>> {self.global_states__vectors}")

                self.episode_prev_states_temp.extend([self.global_states__vectors])

                self.episode_reward += self.reward

                self.episode_current_action.append(self.current_action)
                # self.episode_current_action.append(self.global_actions)

                # If Episode changed
                if self.episode > self.prev_episode:
                    # finding best Episode
                    if self.best_reward < self.episode_reward:
                        self.best_reward = self.episode_reward
                        current_run = path_maker()
                        torch.save(self.agent.policy_net.state_dict(),
                                   os.path.join(current_run, f"best_episode_{self.bot_id}.pt"))
                        with open(os.path.join(current_run, f"best_episode_{self.bot_id}.txt"), "w") as fin:
                            fin.write("")
                        with open(os.path.join(current_run, f"best_episode_{self.bot_id}.txt"), "a") as fin:
                            fin.write(f"================= Module {self.bot_id}\n")
                            fin.write(f"================= Rewards\n")
                            for i in self.episode_reward_temp:
                                fin.write(f"{i}, ")
                            fin.write(f"\n================= States\n")
                            for i in self.episode_states_temp:
                                fin.write(f"{i}, ")
                            fin.write(f"\n================= Actions\n")
                            for i in self.episode_single_action_temp:
                                fin.write(f"{i}, ")
                            fin.write("\n")

                    self.replay_buf_reward.extend([self.episode_reward_temp])
                    self.replay_buf_state.extend([self.episode_states_temp])
                    self.replay_buf_mean_action.extend([self.episode_mean_actions_temp])
                    self.replay_buf_state_.extend([self.episode_prev_states_temp])
                    self.replay_buf_action.extend([self.episode_actions_temp])
                    self.replay_buf_single_action.extend([self.episode_single_action_temp])

                    self.episode_reward_temp = []
                    self.episode_states_temp = []
                    self.episode_mean_actions_temp = []
                    self.episode_prev_states_temp = []
                    self.episode_actions_temp = []
                    self.episode_single_action_temp = []

                    # logger
                    if self.bot_id == LEADER_ID:
                        # logger
                        writer(self.bot_id, NUM_MODULES, self.total_time_elapsed,
                               self.episode_reward, self.loss, self.episode, NN_TYPE)

                    self.episode_reward = 0
                    self.episode_mean_action.clear()
                    self.prev_episode_mean_action.clear()
                    self.episode_current_action.clear()
                    self.prev_episode = self.episode
                    self.batch_ticks += 1
                    self.batch_updated = False
                    self.REPLAY_MEMORY_EPISODE += 1

                # batch is at least at the minimal working size
                if self.batch_ticks >= BATCH_SIZE:
                    # run the NN and collect loss
                    # TODO: do all payload creation here and pass the final to optimizer

                    # if number of passed episodes is less than BUFFER_LIMIT (maximum number of inputs in buffer)
                    if self.episode < BUFFER_LIMIT:
                        # just generating a list form 0-episode
                        episodes = range(self.episode-1)
                    else:
                        # just generating a list from 0-BUFFER_LIMIT
                        episodes = range(BUFFER_LIMIT)

                    # excluding 1 from episodes because the first run has one or more faulty inputs
                    sample = np.random.choice(episodes, BATCH_SIZE-1, replace=False)
                    del episodes

                    state_action_payloads = []
                    expected_state_action_payloads = []

                    for part in sample:
                        number_of_steps = range(len(self.replay_buf_state[part]))
                        for step in number_of_steps:
                            if COMMUNICATION:
                                # adding states
                                robot_state_vectors = [list(itertools.chain(*self.replay_buf_state[part][step]))]
                                # adding previous state
                                robot_state__vectors = [list(itertools.chain(*self.replay_buf_state_[part][step]))]
                                # adding actions
                                robot_state_vectors.append(list(itertools.chain(*self.replay_buf_action[part][step])))
                                robot_state__vectors.append(list(itertools.chain(*self.replay_buf_action[part][step])))
                                # adding mean actions
                                robot_state_vectors.append(list(itertools.chain(*self.replay_buf_mean_action[part][step])))
                                robot_state__vectors.append(list(itertools.chain(*self.replay_buf_mean_action[part][step])))
                                # adding reward
                                robot_state_vectors.append([0, 0, self.replay_buf_reward[part][step]]*NUM_MODULES)
                                robot_state__vectors.append([0, 0, self.replay_buf_reward[part][step]]*NUM_MODULES)
                            else:
                                # adding states
                                robot_state_vectors = list(itertools.chain(*self.replay_buf_state[part][step][self.bot_id-1]))
                                # adding previous state
                                robot_state__vectors = list(itertools.chain(*self.replay_buf_state_[part][step][self.bot_id-1]))
                                # adding actions
                                robot_state_vectors += list(itertools.chain(*self.replay_buf_action[part][step][self.bot_id-1]))
                                robot_state__vectors += list(itertools.chain(*self.replay_buf_action[part][step][self.bot_id-1]))
                                # adding reward
                                robot_state_vectors.append(self.replay_buf_reward[part][step])
                                robot_state__vectors.append(self.replay_buf_reward[part][step])

                            state_action_payloads.append([robot_state_vectors, self.replay_buf_single_action[part][step]])
                            expected_state_action_payloads.append([robot_state__vectors, self.replay_buf_reward[part][step]])

                    self.loss = self.agent.optimize(episode=self.episode, sap=state_action_payloads,
                                                    esap=expected_state_action_payloads)
                    state_action_payloads.clear()
                    expected_state_action_payloads.clear()
                    self.batch_updated = True
                    self.batch_ticks = 0

            # set previous action option to the new one
            self.prev_actions = self.current_action
            if COMMUNICATION:
                # get the array of state vectors
                robot_state_vectors = [list(itertools.chain(*self.global_states_vectors[0:NUM_MODULES])),
                                       list(itertools.chain(*self.global_actions_vectors)),
                                       list(itertools.chain(*self.global_mean_action_vectors)), [0, 0, self.reward]*NUM_MODULES]
            else:
                # get the array of state vectors
                t_action = self.global_states_vectors[0:NUM_MODULES][self.bot_id-1]
                robot_state_vectors = t_action[0:3]
                robot_state_vectors += self.global_actions_vectors[self.bot_id-1]
                robot_state_vectors.append(self.reward)
            try:
                self.current_action = self.agent.choose_action(robot_state_vectors)[0]
            except TypeError:
                print(f"Error with input: {robot_state_vectors} line 547")
                exit(222)

            # if self.bot_id == 1:
            #     print(f"after cleaning global_states__vectors >>> {self.global_states__vectors}")
            # self.global_states__vectors = self.global_states_vectors
            # if self.bot_id == 1:
            #     print(f"after putting new info global_states__vectors >>> {self.global_states__vectors}")
            #
            # self.global_states_vectors = []

            # run the action and change the state
            self.state_changer()
            # notify others of current action
            self.notify_of_an_action(self.current_action)

        else:
            self.act = None
            self.sts = None
            self.old_pos = self.gps.getValues()
            self.t = 0