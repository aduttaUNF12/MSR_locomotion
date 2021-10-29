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
            if COMMUNICATION:
                try:
                    # payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, (9*NUM_MODULES)+1, 1, 1)
                    payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, 18, 1, 1)
                except RuntimeError:
                    # if self.module_number == 1:
                    print(f"temp_ >>> ({temp_}_ ({len(temp_)})")
                    exit(int(f"20{self.module_number}"))
            else:
                payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, (NUM_MODULES*2)+1, 1, 1)
        return payload

    # Greedy action selection
    def choose_action(self, module_actions):
        if random.random() < (1 - self.epsilon):
            payload = self.payload_maker(module_actions)
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

    def optimize(self, episode=None, sap=None, esap=None):
        self.policy_net.optimizer.zero_grad()

        state_action_values = []
        expected_state_action_values = []
        #
        # with open(f"TESTING_FUCKING_OUTPUTS_{self.module_number}", "w") as fin:
        #     fin.write("SAP ======================\n")
        #     for i in sap:
        #         fin.write(f"{i} == ({len(i[0])})\n")
        #     fin.write("ESAP ======================\n")
        #     for e in esap:
        #         fin.write(f"{e} == ({len(e[0])})\n")

        for s in sap:
            # s = [[payload], action]
            # making a payload to send to the NN
            payload = self.payload_maker(s[0])
            # sending payload to the NN
            res = self.policy_net(payload)
            del payload  # garbage collection
            res = res.to('cpu')   # sending res to cpu to clear some VRAM
            state_action_values.append(torch.tensor(res[s[1]],  # getting Q-Value
                                                    dtype=torch.float, requires_grad=True).to('cpu'))
            del res

        for s in esap:
            # s = [[payload], reward]
            # making a payload to send to the NN
            payload = self.payload_maker(s[0])
            # sending payload to the NN
            res = self.target_net(payload).to('cpu').detach()
            del payload  # garbage collection
            # getting index of the largest value in res
            max_index = np.argmax(res)
            expected_state_action_values.append((res[max_index].to('cpu').detach() * self.gamma) + s[1])
            del res

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

