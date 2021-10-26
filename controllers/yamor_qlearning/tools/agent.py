import numpy as np
import itertools
import os
import time
import random
import torch

from .constants import *
from .loggers import path_maker



class Agent:
    # policy_net = does all of the training and tests
    # target_net = final network

    def __init__(self, module_number, number_of_modules, n_actions, lr, gamma=0.99,
                 epsilon=1, eps_dec=1e-5, eps_min=0.01, target_net=None, policy_net=None, nn_type=None):
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
        self.policy_net = policy_net
        # self.target_net = NET_TYPE(self.number_of_modules, self.lr, n_actions)
        self.target_net = target_net
        self.NN_TYPE = nn_type
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def payload_maker(self, temp_):
        if self.NN_TYPE == "FCNN":
            payload = torch.tensor([temp_], dtype=torch.float).to(self.policy_net.device)
        else:
            # payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
            # TODO: regular
            # if self.module_number == 1:
            #     print(f"temp_ >>> {temp_} ({len(temp_)})")
            try:
                payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, (9*NUM_MODULES)+1, 1, 1)
            except RuntimeError:
                if self.module_number == 1:
                    print(f"temp_ >>> ({temp_}_ ({len(temp_)})")
                exit(222)
            # NO COM \/
            # payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 7, 1, 1)
            # for action input
            # payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 3*(NUM_MODULES + 1) + NUM_MODULES, 1, 1)
        return payload

    # Greedy action selection
    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):
            # make list of lists into a single list, and turn it into numpy array
            # TODO: regular
            temp_ = np.array(list(itertools.chain(*module_actions)))
            # temp_ = module_actions
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

    # TODO: document what each buffer contains
    def optimize(self, batch=False, episode_buffer=None, action_buffer=None,
                 reward_buffer=None, vector_states_buffer=None,
                 vector_states__buffer=None, vector_actions_buffer=None,
                 vector_all_mactions_buffer=None, episode=None):
        self.policy_net.optimizer.zero_grad()

        # TODO: make a best episode tracker
        # TODO: matlab basicfitting linear

        # if number of passed episodes is less than BUFFER_LIMIT (maximum number of inputs in buffer)
        if episode < BUFFER_LIMIT:
            # just generating a list form 0-episode
            episodes = range(episode-1)
        else:
            # just generating a list from 0-BUFFER_LIMIT
            episodes = range(BUFFER_LIMIT)
        # excluding 1 from episodes because the first run has one or more faulty inputs
        sample = np.random.choice(episodes, BATCH_SIZE-1, replace=False)
        del episodes

        state_action_values = []
        expected_state_action_values = []

        # sample (list) contains indexes of Episodes which are used in training
        for part in sample:
            number_of_steps = range(len(vector_states_buffer[part])-1)
            for step in number_of_steps:
                # turning all of the lists of lists into one big list
                # adding states
                robot_state_vectors = list(itertools.chain(*vector_states_buffer[part][step]))
                # adding actions
                robot_state_vectors += list(itertools.chain(*vector_actions_buffer[part][step]))
                # adding mean actions
                robot_state_vectors += list(itertools.chain(*vector_all_mactions_buffer[part][step]))
                # adding reward
                robot_state_vectors.append(reward_buffer[part][step])
                # making a payload to send to the NN
                payload = self.payload_maker(robot_state_vectors)
                # sending payload to the NN
                res = self.policy_net(payload)
                del payload  # garbage collection
                robot_state_vectors.clear()   # garbage collection
                res = res.to('cpu')   # sending res to cpu to clear some VRAM
                action = action_buffer[part][step]  # getting action
                state_action_values.append(torch.tensor(res[action],  # getting Q-Value
                                                        dtype=torch.float, requires_grad=True).to('cpu'))
                del action, res

        for part in sample:
            number_of_steps = range(len(vector_states__buffer[part])-1)
            for step in number_of_steps:
                # turning all of the lists of lists into one big list
                # adding previous states states
                robot_state_vectors = list(itertools.chain(*vector_states__buffer[part][step]))
                # adding actions
                robot_state_vectors += list(itertools.chain(*vector_actions_buffer[part][step]))
                # adding mean actions
                robot_state_vectors += list(itertools.chain(*vector_all_mactions_buffer[part][step]))
                # adding reward
                robot_state_vectors.append(reward_buffer[part][step])
                # making a payload to send to the NN
                payload = self.payload_maker(robot_state_vectors)
                # sending payload to the NN
                res = self.target_net(payload).to('cpu').detach()
                del payload  # garbage collection
                robot_state_vectors.clear()   # garbage collection
                # getting index of the largest value in res
                max_index = np.argmax(res)
                expected_state_action_values.append((res[max_index].to('cpu').detach() * self.gamma) + reward_buffer[part][step])
                del res

        del sample

        state_action_values = torch.stack(state_action_values)
        state_action_values = state_action_values.double().float()

        # if self.module_number == 1:
        #     print(f"input ev >>> {t}\nout values >>> {state_action_values}")

        # expected_state_action_values = torch.stack(expected_state_action_values).double().float()
        expected_state_action_values = np.stack(expected_state_action_values)

        # if self.module_number == 1:
        #     print(f"input values >>> {t}\nout values >>> {expected_state_action_values}")

        # state_action_values = torch.tensor(state_action_values).to(self.target_net.device)
        expected_state_action_values = torch.tensor(expected_state_action_values).to(self.target_net.device)
        # loss = self.policy_net.loss(state_action_values, expected_state_action_values)
        loss = self.policy_net.loss(state_action_values.to(self.target_net.device), expected_state_action_values.double().float())
        del expected_state_action_values, state_action_values
        # TODO figure out why loss is astronomical
        # TODO: run for 20k episodes for current config
        loss.backward()
        self.policy_net.optimizer.step()
        self.decrement_epsilon()
        if True:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{episode}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

            current_run = path_maker()
            torch.save(self.target_net.state_dict(), os.path.join(current_run, "agent.pt"))

            self.updated = True
        elif EPISODE % UPDATE_PERIOD != 0:
            self.updated = False

        return loss

