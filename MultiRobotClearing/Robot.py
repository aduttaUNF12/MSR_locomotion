import math
import torch
import random
import logging



class Epsilon():
    def __init__(self, start, end, decay):
        self.start = start #1
        self.end = end #0.01
        self.decay = decay #0.999
        self.epsilon = start
        
        # not being used anymore
    def choose_action(self, time_step):
        self.epsilon = self.end + (self.start - self.end) * math.exp(-1. * (time_step )/ self.decay)
        return self.epsilon

class Robot():
    def __init__(self):
        self.x_coordinate = 0
        self.y_coordinate = 0
        self.steps = 0
        #self.action_space=7
        #self.decay = 2000
        #self.steps_given = 100
        #self.model_count = 0
        #self.random_count = 0
        #self.rate = Epsilon(1, 0.01, self.decay)
        #change it to exponential decay, right now it is at a fixed rate
    
    
    def select_action(self, state, policy_network, environment, epsilon, mean_action=None):

        
        #self.decay = self.decay * .9999
    
        self.steps += 1

        #self.steps_given -= 1
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
            self.model_count += 1
            action = torch.argmax(temp_q_values.flatten()).view(1, 1)
            return action, epsilon