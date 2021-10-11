import numpy as np
import itertools
import os
import time
import random
import torch

from .constants import *



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
            payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 3*(NUM_MODULES + 1), 1, 1)
            # for action input
            # payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 3*(NUM_MODULES + 1) + NUM_MODULES, 1, 1)
        return payload

    # Greedy action selection
    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):
            # make list of lists into a single list, and turn it into numpy array
            temp_ = np.array(list(itertools.chain(*module_actions)))
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

    def optimize(self, batch=False, episode_buffer=None, action_buffer=None, reward_buffer=None, vector_states_buffer=None,
                 vector_mactions_buffer=None, vector_states__buffer=None, vector_mactions__buffer=None,
                 vector_actions_buffer=None, vector_actions__buffer=None, episode=None):
        self.policy_net.optimizer.zero_grad()

        # get the current amount of Episodes
        episodes = range(len(episode_buffer))[1:]
        # Not necessary, but was used to testing running optimizer after every Episode as well as a batch (NOT USED)
        if len(episode_buffer) >= BATCH_SIZE and batch is True:
            sample = np.random.choice(episodes, BATCH_SIZE-1, replace=False)
        else:
            sample = np.array([len(episode_buffer)])
        del episodes

        ranges = []

        for s in sample:
            # This is done mainly because all of the Episodes contain 577 actions, but some have 576
            # (it might just be the first Episode that only has 576 but I need to look further into it,
            # for now this works)
            if episode_buffer[s]['max'] - episode_buffer[s]['min'] > 576:
                sub = episode_buffer[s]['max'] - episode_buffer[s]['min'] - 576
                episode_buffer[s]['max'] = episode_buffer[s]['max'] - sub
            # make a list of numbers from min to max and add that list to ranges list (each list contains
            #   actions taken during the Episode
            ranges.append(np.arange(episode_buffer[s]['min'],
                                    episode_buffer[s]['max']))

        del sample

        state_action_values = []
        expected_state_action_values = []
        # iterates over the array of range arrays,  index1 is the id of the range array, r is the array or range
        for index1, r in enumerate(ranges):
            # pull values corresponding to the r range
            temp_action = action_buffer.get(r)
            temp_rewards = reward_buffer.get(r)

            # iterates over the r, index2 is the id of the item in r, and item is the entry in the list
            for index2, item in enumerate(r):
                # get the list of state vectors
                robot_state_vectors = vector_states_buffer[item][0:NUM_MODULES]
                # add corresponding action
                # t = vector_actions_buffer[item][self.module_number - 1]
                # robot_state_vectors.append(t)

                # add corresponding mean action
                robot_state_vectors.append(vector_mactions_buffer[item])
                # convert a list of lists into a single list
                temp_ = np.array(list(itertools.chain(*robot_state_vectors)))
                payload = self.payload_maker(temp_)
                del temp_, robot_state_vectors
                res = self.policy_net(payload)
                del payload
                res = res.to('cpu')

            # res[temp_action[index2]] = select the estimate value form the res list which corresponds to an
                #   action of at the same index; adds the value in tensor form to the state_action_values
                state_action_values.append(torch.tensor(res[temp_action[index2]], dtype=torch.float, requires_grad=True).to('cpu'))
                # state_action_values.append(res[temp_action[index2]].to('cpu').detach().numpy())
                del res
            del index2, item

            for index3, item in enumerate(r):
                # get the list of state vectors
                robot_state_vectors_ = vector_states__buffer[item][0:NUM_MODULES]
                # add corresponding action
                # t = vector_actions__buffer[item][self.module_number - 1]
                # robot_state_vectors_.append(t)
                # add corresponding mean action
                robot_state_vectors_.append(vector_mactions__buffer[item])
                # convert a list of lists into a single list
                temp_ = np.array(list(itertools.chain(*robot_state_vectors_)))
                del robot_state_vectors_
                payload = self.payload_maker(temp_)
                del temp_
                res = self.target_net(payload).to('cpu').detach().numpy()
                del payload
                # get the position of the largest estimate in the res list, multiplies it by gamma and
                #   adds reward associated with this action in a given Episode

                max_index = np.argmax(res)
                # expected_state_action_values.append((res[max_index].to(self.target_net.device) * self.gamma) + temp_rewards[index3])
                expected_state_action_values.append((res[max_index] * self.gamma) + temp_rewards[index3])

                # expected_state_action_values.append((torch.tensor(np.argmax(res.to('cpu').detach().numpy()), dtype=torch.float).to(self.target_net.device) * self.gamma) + temp_rewards[index3])
                del res
            del index3, item
        state_action_values = torch.stack(state_action_values)
        state_action_values = state_action_values.double().float()

        # expected_state_action_values = torch.stack(expected_state_action_values).double().float()
        expected_state_action_values = np.stack(expected_state_action_values)
        # state_action_values = torch.tensor(state_action_values).to(self.target_net.device)
        expected_state_action_values = torch.tensor(expected_state_action_values).to(self.target_net.device)
        # loss = self.policy_net.loss(state_action_values, expected_state_action_values)
        loss = self.policy_net.loss(state_action_values.to(self.target_net.device), expected_state_action_values.double().float())
        del expected_state_action_values, state_action_values
        loss.backward()
        self.policy_net.optimizer.step()
        self.decrement_epsilon()
        if True:
            # if EPISODE % UPDATE_PERIOD == 0 and self.updated is False:
            # if self.updated is False:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{episode}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

            if BASE_LOGS_FOLDER is not None:
                torch.save(self.target_net.state_dict(), os.path.join(BASE_LOGS_FOLDER, "agent.pt"))

            self.updated = True
        elif EPISODE % UPDATE_PERIOD != 0:
            self.updated = False

        return loss

