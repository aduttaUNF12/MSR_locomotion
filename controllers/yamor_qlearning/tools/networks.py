import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .constants import NUM_MODULES, CEIL_MODE


# the NN itself
class CNN(nn.Module):
    def __init__(self, lr, n_actions):
        super(CNN, self).__init__()
        self.number_of_modules = NUM_MODULES
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device
        # useful? https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.number_of_modules, kernel_size=(1, 3), stride=(1, 3)).to(self.device)
        # ceil mode needs to be true or else a 0 column is added to both sides
        self.p1 = nn.AvgPool1d(3, ceil_mode=CEIL_MODE)

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=self.number_of_modules**2, kernel_size=(1, 3), stride=(1, 3)).to(self.device)
        self.p2 = nn.AvgPool1d(3, ceil_mode=CEIL_MODE)

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=self.number_of_modules**3, kernel_size=(1, 3), stride=(1, 3)).to(self.device)
        self.p3 = nn.AvgPool1d(3, ceil_mode=CEIL_MODE)

        # with kernel (1, 3) we should be combining one vector at a time, in bits of 3
        x = torch.tensor([[0., 0., 1., 1., 0., 0., 1., 0., 0.],  # all states
                          [0., 0., 1., 1., 0., 0., 1., 0., 0.],  # all actions
                          [0., 0., 1., 1., 0., 0., 1., 0., 0.],  # all mean action
                          [0., 0., 1.25, 0., 0., 1.25, 0., 0., 1.25]  # reward
                          ], dtype=torch.float32).to(self.device).view(4, 1, 1, 9)
        # print(f"x >>> {x} (shape) {x.shape}")
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self.number_of_modules**2, self.number_of_modules**2).to(self.device)
        self.fc2 = nn.Linear(self.number_of_modules**2, 3).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def convs(self, x, dbg=False):
        x = self.conv1(x)
        if dbg:
            print(f"conv1 >>> {x} (shape) {x.shape}")
            print(f"conv1 reshape >>> {x.view(12, 1, 3)} (shape) {x.view(12, 1, 3).shape}")
        x = x.view(12, 1, 3)
        # x = self.bn1(x)
        x = self.p1(x)
        if dbg:
            print(f"bn1 >>> {x} (shape) {x.shape}")
        x = F.relu(x)
        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
            print(f"relu reshape >>> {x.view(4, 1, 1, 3)} (shape) {x.view(4, 1, 1, 3).shape}")
        x = x.view(4, 1, 1, 3)

        x = self.conv2(x)
        if dbg:
            print(f"conv2 >>> {x} (shape) {x.shape}")
            print(f"conv2 reshape >>> {x.view(4, 1, self.number_of_modules**2)}"
                  f" (shape) {x.view(4, 1, self.number_of_modules**2).shape}")
        x = x.view(4, 1, 9)
        # x = self.bn2(x)
        x = self.p2(x)
        if dbg:
            print(f"bn2 >>> {x} (shape) {x.shape}")
        x = F.relu(x)

        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
            print(f"relu reshape >>> {x.view(4, 1, 1, 3)} (shape) {x.view(4, 1, 1, 3).shape}")
        x = x.view(4, 1, 1, 3)
        x = self.conv3(x)
        if dbg:
            print(f"conv3 >>> {x} (shape) {x.shape}")
            print(f"conv3 reshape >>> {x.view(4, 1, self.number_of_modules**3)}"
                  f" (shape) {x.view(4, 1, self.number_of_modules**3).shape}")
        x = x.view(4, 1, self.number_of_modules**3)
        # x = self.bn3(x)
        x = self.p3(x)
        if dbg:
            print(f"bn3 >>> {x}(shape) {x.shape}")
        x = F.relu(x)
        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
            print(f"relu reshape >>> {x.view(4, 1, 1, self.number_of_modules**2)}"
                  f" (shape) {x.view(4, 1, 1, self.number_of_modules**2).shape}")
        x = x.view(4, 1, 1, self.number_of_modules**2)

        if self._to_linear is None:
            # self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            self._to_linear = x
            if dbg:
                print(f"_to_linear >>> {x} (shape) {x.shape}")
        return x

    def forward(self, actions, dbg=False):
        if dbg:
            print(f"################\nactions >>> {actions} (shape) {actions.shape}")
        x = self.convs(actions, dbg=dbg)
        if dbg:
            print(f"convs >>> {x} (shape) {x.shape}")
        x = x.view(4, self.number_of_modules**2)
        # x = x.view(6, 6)
        if dbg:
            print(f"view >>> {x} (shape) {x.shape}")
        x = self.fc1(x)
        if dbg:
            print(f"fc1 >>> {x} (shape) {x.shape}")
        x = F.relu(x)
        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
        x = self.fc2(x)
        if dbg:
            print(f"fc2 >>> {x} (shape) {x.shape}")

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
