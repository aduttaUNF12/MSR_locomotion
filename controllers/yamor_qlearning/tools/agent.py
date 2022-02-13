import os
import random
import time

import numpy as np
import torch

from .constants import *
from .loggers import path_maker


class Agent:
    # policy_net = does all of the training and tests
    # target_net = final network

    def __init__(self, module_number, number_of_modules, n_actions=None, gamma=None,
                 epsilon=None, eps_dec=None, eps_min=None, target_net=None, policy_net=None, nn_type=None):
        self.module_number = module_number   # only used for logging
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
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
            payload = torch.tensor(temp_, dtype=torch.float).to(self.policy_net.device)
        else:
            if COMMUNICATION:
                try:
                    if FIX:
                        # fix
                        payload = torch.tensor(temp_, dtype=torch.float).to(self.policy_net.device)
                    else:
                        # original
                        payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, (9*NUM_MODULES)+1, 1, 1)
                except RuntimeError:
                    # if self.module_number == 1:
                    print(f"temp_ >>> ({temp_}_ ({len(temp_)})")
                    exit(int(f"6{self.module_number}"))
            else:
                payload = torch.tensor([temp_]*32, dtype=torch.float).to(self.policy_net.device).view(32, (NUM_MODULES*2)+1, 1, 1)
        return payload

    # Greedy action selection
    def choose_action(self, module_actions, episode):
        sample = random.random()
        # FOR DECAY AT EACH ACTION
        # self.decrement_epsilon()
        if episode > DECAY_PAUSE_EPISODE:
            if EPS_EXP:
                eps_threshold = self.eps_min + (self.epsilon - self.eps_min) * math.exp(-1 * (episode - DECAY_PAUSE_EPISODE)/ self.eps_dec)
                res = sample > eps_threshold

                if self.module_number == LEADER_ID:
                    with open("log.txt", "a") as fout:
                        fout.write(f"################### In Choose Action (EXP) ###################\n")
                        fout.write(f"EPS threshold {eps_threshold}\n")

            else:
                eps_threshold = (1 - self.epsilon)
                res = sample < eps_threshold
            if res:
                payload = self.payload_maker(module_actions)
                action = self.policy_net.forward(payload)

                if self.module_number == LEADER_ID:
                    with open("log.txt", "a") as fout:
                        fout.write(f"################### Action From NN ###################\n")
                        fout.write(f"Module Actions {module_actions}\n")
                        fout.write(f"Payload {payload}\n")
                        fout.write(f"Actions {action}\n")
                        fout.write(f"Argmax Actions {[np.argmax(action.to('cpu').detach().numpy())]}\n")

                return [np.argmax(action.to('cpu').detach().numpy())]
            else:
                action = np.random.choice(self.action_space, 1)

                # if self.module_number == LEADER_ID:
                #     with open("log.txt", "a") as fout:
                #         fout.write(f"################### Action Random ###################\n")
                #         fout.write(f"Actions {action}\n")

            return action
        else:
            action = np.random.choice(self.action_space, 1)

            # if self.module_number == LEADER_ID:
            #     with open("log.txt", "a") as fout:
            #         fout.write(f"################### Action 300 Init Eps ###################\n")
            #         fout.write(f"Actions {action}\n")

            return action

    def decrement_epsilon(self):
        # decrease epsilon by set value if epsilon != min epsilon
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def optimize(self, episode=None, sap=None, esap=None, step=None):

        state_action_values = []
        expected_state_action_values = []

        # print(f"SAP >>> {sap}")
        # print(f"SAP[0] >>> {sap[0]}")
        # print(f"SAP[1] >>> {sap[1]}")
        # exit(111)
        # for s in sap:
        #     for i, sa in enumerate(s[0]):
        #         payload = self.payload_maker(sa)
        #         # sending payload to the NN
        #         res = self.policy_net(payload)
        #         del payload  # garbage collection
        #         res = res.to('cpu')   # sending res to cpu to clear some VRAM
        #         state_action_values.append(torch.tensor(res[s[1][i]],  # getting Q-Value
        #                                             dtype=torch.float, requires_grad=True).to('cpu'))
        #     del res

        # print(f"sap >>>> {sap}")
        if self.module_number == LEADER_ID:
            with open("log.txt", "a") as fout:
                fout.write(f"################### Optimizer ###################\n")
                fout.write(f"################### SAP ###################\n")
                # fout.write(f"{sap[0:32]}\n")
                for thing in sap[0:32]:
                    fout.write(f"{thing}\n")
                fout.write(f"################### ESAP ###################\n")
                # fout.write(f"{esap[0:32]}\n")
                for thing in esap[0:32]:
                    fout.write(f"{thing}\n")
        for i, s in enumerate(sap):
            # exit(111)
            payload = self.payload_maker(s[0])
            # sending payload to the NN
            res = self.policy_net(payload)
            res = res.to('cpu')   # sending res to cpu to clear some VRAM

            if self.module_number == LEADER_ID:
                if i < 32:
                    with open("log.txt", "a") as fout:
                        fout.write(f"################### SAP ({i}) ###################\n")
                        fout.write(f"################### Payload ###################\n")
                        fout.write(f"{payload}\n")
                        fout.write(f"################### Res ###################\n")
                        fout.write(f"{res}\n")
                        fout.write(f"################### State Action Value ###################\n")
                        fout.write(f"{res[s[1]]}\n")

            del payload  # garbage collection
            state_action_values.append(torch.tensor(res[s[1]],  # getting Q-Value
                                                    dtype=torch.float, requires_grad=True).to('cpu'))
            del res

        for i, s in enumerate(esap):
            payload = self.payload_maker(s[0])
            # sending payload to the NN
            res = self.target_net(payload).to('cpu').detach()
            max_index = np.argmax(res)
            if self.module_number == LEADER_ID:
                if i < 32:
                    with open("log.txt", "a") as fout:
                        fout.write(f"################### ESAP ({i}) ###################\n")
                        fout.write(f"################### Payload ###################\n")
                        fout.write(f"{payload}\n")
                        fout.write(f"{res}\n")
                        fout.write(f"################### Max Index ###################\n")
                        fout.write(f"{max_index}\n")
                        fout.write(f"################### Expected State Action Value ###################\n")
                        fout.write(f"{(res[max_index].to('cpu').detach() * self.gamma) + s[1]}\n")

            del payload  # garbage collection
            expected_state_action_values.append((res[max_index].to('cpu').detach() * self.gamma) + s[1])
            del res

        state_action_values = torch.stack(state_action_values)
        state_action_values = state_action_values.double().float()

        # expected_state_action_values = torch.stack(expected_state_action_values).double().float()
        expected_state_action_values = np.stack(expected_state_action_values)
        # state_action_values = torch.tensor(state_action_values).to(self.target_net.device)
        expected_state_action_values = torch.tensor(expected_state_action_values).to(self.target_net.device)

        if self.module_number == LEADER_ID:
            with open("log.txt", "a") as fout:
                fout.write(f"################### State Action Values ###################\n")
                # fout.write(f"{state_action_values[0:32]}\n")
                for thing in state_action_values[0:32]:
                    fout.write(f"{thing}\n")
                fout.write(f"################### Expected State Action Values ###################\n")
                # fout.write(f"{expected_state_action_values[0:32]}\n")
                for thing in expected_state_action_values[0:32]:
                    fout.write(f"{thing}\n")

        # loss = self.policy_net.loss(state_action_values, expected_state_action_values)
        self.policy_net.optimizer.zero_grad()
        loss = self.policy_net.loss(state_action_values.to(self.policy_net.device), expected_state_action_values.double().float())

        if str(loss) is "nan":
            if self.module_number == LEADER_ID:
                with open("loss_log.txt", 'a') as fout:
                    fout.write("################### State Action Values ###################\n")
                    fout.write(f"{state_action_values}\n")
                    fout.write("################### Expected State Action Values ###################\n")
                    fout.write(f"{expected_state_action_values}\n")
                    fout.write("################### Loss ###################\n")
                    fout.write(f"{loss}\n")

        if self.module_number == LEADER_ID:
            with open("log.txt", "a") as fout:
                fout.write(f"################### LOSS ###################\n")
                fout.write(f"{loss}\n")

        del expected_state_action_values, state_action_values
        # TODO figure out why loss is astronomical
        loss.backward()
        self.policy_net.optimizer.step()
        if not EPS_EXP:
            # original
            self.decrement_epsilon()
            # pass

        if step >= BATCH_SIZE and step % UPDATE_PERIOD == 0:
            print(f"Updated model: {time.strftime('%H:%M:%S', time.localtime())} ============== Episode:{episode}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

            current_run = path_maker()
            torch.save(self.target_net.state_dict(), os.path.join(current_run, "agent.pt"))

        return loss

