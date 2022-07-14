import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, gpu):
        super(Model, self).__init__()
        self.random_actions = 0
        self.model_actions = 0
        self.n_actions = 7
        self.embedding_size = 8
        self.width = 16
        self.height = 16
        self.to(gpu)
        self.embedder = nn.Linear(self.n_actions, self.embedding_size).to(gpu)
        self.observation_layer = nn.Conv2d(3, 16, kernel_size=5, stride=1).to(gpu)
        self.norm1 = nn.BatchNorm2d(8).to(gpu)
        self.observation_layer2 = nn.Conv2d(16, 16, kernel_size=5, stride=1).to(gpu)
        self.norm2 = nn.BatchNorm2d(8).to(gpu)
        #self.observation_layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        #self.observation_layer3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
       #

        def convert_to_size(size, kernel_size=5, stride=1):
            new_size = (size - (kernel_size - 1) - 1) // stride + 1
            return new_size
        

        self.width = convert_to_size(convert_to_size(self.width))
        self.height = convert_to_size(convert_to_size(self.height))
        #width = convert_to_size(convert_to_size(convert_to_size(self.width)))
        #height = convert_to_size(convert_to_size(convert_to_size(self.height)))

        self.linear_layer1 = nn.Linear(16 * self.width * self.height + 1, 256).to(gpu)
        self.linear_layer2 = nn.Linear(256, 256).to(gpu)
        self.linear_layer3 = nn.Linear(256, 256).to(gpu)
        self.output_layer = nn.Linear(256, 5).to(gpu)

    def forward(self, observation, budget):
        #action_embedded = self.embedder(action)
        budget = budget.to(device)
        observation = observation.to(device)
        observation = torch.relu(self.observation_layer(observation.double()))
        observation = torch.relu(self.observation_layer2(observation))
        observation = observation.view(observation.size(0), -1)
        observation_budget = torch.cat([observation, budget], dim=-1)
        observation_budget = torch.relu(self.linear_layer1(observation_budget))
        observation_budget = torch.relu(self.linear_layer2(observation_budget))
        observation_budget = torch.relu(self.linear_layer3(observation_budget))

        #observation = torch.tanh(self.observation_layer2(observation))
        #observation = torch.tanh(self.observation_layer3(observation))
        #observation = observation.view(observation.size(0), 1, -1)
        #lstm_input = torch.cat([observation.view(observation.size(0), 1, -1), action_embedded.view(action_embedded.size(1), action_embedded.size(0), action_embedded.size(2))], dim=-1)
        #
        #if hidden is not None:
        #lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        #else:
        #lstm_out, hidden_out = self.lstm(lstm_input)

        #
        q_values = self.output_layer(observation_budget)
        return q_values

    
    