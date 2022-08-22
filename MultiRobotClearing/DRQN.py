import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_actions = 0
        self.model_actions = 0
        self.n_actions = 5
        self.embedding_size = 8
        self.width = 16
        self.height = 16
        self.to(self.device)
        self.embedder = nn.Linear(self.n_actions, self.embedding_size).to(self.device)
        self.observation_layer = nn.Conv2d(4, 16, kernel_size=5, stride=1).to(self.device)
        self.norm1 = nn.BatchNorm2d(8).to(self.device)
        self.observation_layer2 = nn.Conv2d(16, 16, kernel_size=5, stride=1).to(self.device)
        self.norm2 = nn.BatchNorm2d(8).to(self.device)

        def convert_to_size(size, kernel_size=5, stride=1):
            new_size = (size - (kernel_size - 1) - 1) // stride + 1
            return new_size

        self.width = convert_to_size(convert_to_size(self.width))
        self.height = convert_to_size(convert_to_size(self.height))

        self.linear_layer1 = nn.Linear(16 * self.width * self.height + 1, 256).to(self.device)
        self.linear_layer2 = nn.Linear(256, 256).to(self.device)
        self.linear_layer3 = nn.Linear(256, 256).to(self.device)
        self.output_layer = nn.Linear(256, 5).to(self.device)

    def forward(self, observation, mean_action):
        mean_action = mean_action.to(self.device)
        observation = observation.to(self.device)
        observation = torch.relu(self.observation_layer(observation.double()))
        observation = torch.relu(self.observation_layer2(observation))
        observation = observation.view(observation.size(0), -1)
        observation_budget = torch.cat([observation, mean_action], dim=-1)
        observation_budget = torch.relu(self.linear_layer1(observation_budget))
        observation_budget = torch.relu(self.linear_layer2(observation_budget))
        observation_budget = torch.relu(self.linear_layer3(observation_budget))

        q_values = self.output_layer(observation_budget)
        return q_values


