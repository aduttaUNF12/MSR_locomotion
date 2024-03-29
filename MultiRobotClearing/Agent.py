import math
import torch
import random
import logging


class Epsilon():
    def __init__(self, start, end, decay):
        self.start = start #1
        self.end = end  #0.01
        self.decay = decay #0.999
        self.epsilon = start

        # not being used anymore
    def choose_action(self, time_step):
        self.epsilon = self.end + (self.start - self.end) * math.exp(-1. * (time_step )/ self.decay)
        return self.epsilon


class Agent():
    def __init__(self, a_id):
        """

        :param a_id: identifier of the agent
        """
        self.agent_id = a_id
        # current x and y
        self.x_coordinate = 0
        self.y_coordinate = 0
        # number of total steps taken
        self.steps = 0
        # previous x and y
        self.previous_x = 0
        self.previous_y = 0
        # times robot stood still
        self.stay_count = 0

    def select_action(self, state, policy_network, environment, epsilon, mean_action=None):
        self.steps += 1

        with torch.no_grad():
            q_values = policy_network(state, mean_action)
            temp_q_values = environment.check_action(action=q_values, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate)
        if epsilon > random.random():
            while True:
                action = torch.randint(0, 5, (1,)).item()
                if temp_q_values.flatten()[action] != float("-inf"):
                    break
            return torch.tensor([[action]]), epsilon
        else:
            action = torch.argmax(temp_q_values.flatten()).view(1, 1)
            return action, epsilon