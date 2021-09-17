import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# the NN itself
class CNN(nn.Module):
    def __init__(self, number_of_modules, lr, n_actions):
        super(CNN, self).__init__()
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        #  TODO IMPORTANT: 3*(number_of_modules + 1) + (number_of_modules * 3), + NUM MODULES * 3 is the only way to get payload states+action+mean
        self.conv1 = nn.Conv2d(in_channels=3*(number_of_modules + 1) + (number_of_modules * 3), out_channels=32, kernel_size=(1, 1)).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)).to(self.device)
        self.bn2 = nn.BatchNorm2d(64).to(self.device)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)).to(self.device)
        self.bn3 = nn.BatchNorm2d(128).to(self.device)

        x = torch.rand((32, 3*(number_of_modules + 1) + (number_of_modules * 3))).to(self.device).view(32, 3*(number_of_modules + 1) + (number_of_modules * 3), 1, 1)

        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 515).to(self.device)
        self.fc2 = nn.Linear(515, 3).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def convs(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, actions):
        x = self.convs(actions)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x[0]


class FCNN(nn.Module):
    def __init__(self, number_of_modules, lr, n_actions):
        super(FCNN, self).__init__()
        self.number_of_modules = number_of_modules
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        self.fc1 = nn.Linear(3*(number_of_modules + 1), 32).to(self.device)
        self.fc2 = nn.Linear(32, 64).to(self.device)
        self.fc3 = nn.Linear(64, self.n_actions).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, actions):
        x = F.relu(self.fc1(actions))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x[0]
