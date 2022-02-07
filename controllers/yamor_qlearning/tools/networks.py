import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .constants import COMMUNICATION, NUM_MODULES, FIX


# the NN itself
class CNN(nn.Module):
    def __init__(self, lr, n_actions):
        super(CNN, self).__init__()
        self.number_of_modules = NUM_MODULES
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        if COMMUNICATION:
            # original
            if not FIX:
                self.conv1 = nn.Conv2d(in_channels=(9*self.number_of_modules)+1, out_channels=32, kernel_size=(1, 1)).to(self.device)
            else:
                # fix (possibly)
                # self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.number_of_modules, kernel_size=(1, 1)).to(self.device)
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(1, 3)).to(self.device)
                # 3 in_channels each one representing as a vector (remove reward) only state, action, ma
                #   16 out_channels (seems paper)
        else:
            self.conv1 = nn.Conv2d(in_channels=(self.number_of_modules*2)+1, out_channels=32, kernel_size=(1, 1)).to(self.device)
        # original
        if not FIX:
            self.bn1 = nn.BatchNorm2d(32, affine=False).to(self.device)
        else:
            # fix
            # self.bn1 = nn.BatchNorm2d(self.number_of_modules, affine=False).to(self.device)
            # self.bn1 = nn.BatchNorm2d(14, affine=False).to(self.device)
            self.bn1 = nn.BatchNorm2d(7, affine=False).to(self.device)
            # self.bn1 = nn.BatchNorm1d(7, affine=False).to(self.device)

        # original
        if not FIX:
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)).to(self.device)
        else:
            # fix
            # self.conv2 = nn.Conv2d(in_channels=self.number_of_modules, out_channels=self.number_of_modules*2, kernel_size=(1, 1)).to(self.device)
            self.conv2 = nn.Conv2d(in_channels=7, out_channels=5, kernel_size=(1, 1)).to(self.device)
            # 16 in_channels each one representing as a vector (remove reward) only state, action, ma
            #   32 out_channels (seems paper)

        # original
        if not FIX:
            self.bn2 = nn.BatchNorm2d(64, affine=False).to(self.device)
        else:
            # fix
            # self.bn2 = nn.BatchNorm2d(self.number_of_modules*2, affine=False).to(self.device)
            self.bn2 = nn.BatchNorm2d(5, affine=False).to(self.device)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)).to(self.device)
        # self.conv3 = nn.Conv2d(in_channels=self.number_of_modules*2, out_channels=self.number_of_modules*9, kernel_size=(1, 1)).to(self.device)
        # self.bn3 = nn.BatchNorm2d(128, affine=False).to(self.device)
        # self.bn3 = nn.BatchNorm2d(self.number_of_modules*9, affine=False).to(self.device)

        # Input for the action input
        # x = torch.rand((32, 3*(number_of_modules + 1) + (number_of_modules))).to(self.device).view(32, 3*(number_of_modules + 1) + (number_of_modules), 1, 1)

        # x = torch.rand((32, 3*(number_of_modules + 1))).to(self.device).view(32, 3*(number_of_modules + 1), 1, 1)
        # TODO: regular
        if COMMUNICATION:
            # original
            if not FIX:
                x = torch.rand((32, (9*self.number_of_modules)+1)).to(self.device).view(32, (9*self.number_of_modules)+1, 1, 1)
            else:
                # self.number_of_modules * 3 since each module has a 3 element vector
                x = torch.rand((3, self.number_of_modules*3)).to(self.device).view(1, 3, 1, self.number_of_modules*3)
        else:
            x = torch.rand((32, (self.number_of_modules*2)+1)).to(self.device).view(32, (self.number_of_modules*2)+1, 1, 1)

        self._to_linear = 10
        self.convs(x, True)

        # self.fc1 = nn.Linear(self._to_linear, 515).to(self.device)

        # in and out 32
        self.fc1 = nn.Linear(self._to_linear, self._to_linear).to(self.device)
        # self.fc2 = nn.Linear(515, 3).to(self.device)

        self.fc2 = nn.Linear(self._to_linear, 3).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def convs(self, x, dbg=None):
        if dbg:
            print(f"In convs()")
            print(f"x >>> {x}  (shape) {x.shape}")
        x = self.conv1(x)
        if dbg:
            print(f"conv1 >>> {x} (shape) {x.shape}")
        x = self.bn1(x)
        if dbg:
            print(f"bn1 >>> {x} (shape) {x.shape}")
        x = F.relu(x)
        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
        x = self.conv2(x)
        if dbg:
            print(f"conv2 >>> {x} (shape) {x.shape}")
        x = self.bn2(x)
        if dbg:
            print(f"bn2 >>> {x} (shape) {x.shape}")
        x = F.relu(x)
        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
        # x = self.conv3(x)
        # if dbg:
        #     print(f"conv3 >>> {x} (shape) {x.shape}")
        # x = self.bn3(x)
        # if dbg:
        #     print(f"bn3 >>> {x} (shape) {x.shape}")
        # x = F.relu(x)
        # if dbg:
        #     print(f"relu >>> {x} (shape) {x.shape}")

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            if dbg:
                print(f"_to_linear >>> {self._to_linear}")

        return x

    def forward(self, actions, dbg=None):
        if dbg:
            print(f"################\nactions >>> {actions} (shape) {actions.shape}")
        x = self.convs(actions, dbg=dbg)
        if dbg:
            print(f"convs >>> {x} (shape) {x.shape}")
        x = x.view(x.size(0), -1)
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
    def __init__(self, lr, n_actions):
        super(FCNN, self).__init__()
        self.number_of_modules = NUM_MODULES
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # send network to device

        # all states, all action, all mean actions, my reward (so +1)
        if COMMUNICATION:
            # self.number_of_modules*9 since each module has 3 input channels and each channel
            #   has a relating 3 piece vector
            # self.fc1 = nn.Linear(self.number_of_modules*9, self.number_of_modules*9).to(self.device)
            self.fc1 = nn.Linear(self.number_of_modules*9, 32).to(self.device)
        else:
            self.fc1 = nn.Linear((2*self.number_of_modules) + 1, 32).to(self.device)

        # self.fc2 = nn.Linear(self.number_of_modules*9, self.number_of_modules*9).to(self.device)
        self.fc2 = nn.Linear(32, 64).to(self.device)
        # self.fc3 = nn.Linear(self.number_of_modules*9, 3).to(self.device)
        self.fc3 = nn.Linear(64, 3).to(self.device)
        # TODO: add another fc layer with (self.number_of_modules*9)/9, 3 set up

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, actions):
        # print(f"actions >>> {actions} (shape) {actions.shape}")
        x = self.fc1(actions)
        # print(f"fc1 done")
        x = F.relu(x)
        x = self.fc2(x)
        # print(f"fc2 done")
        x = F.relu(x)
        x = self.fc3(x)
        return x
