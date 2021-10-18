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
        # 1d [32, 12, 3]
        # 2d [32, 12, 3, 3]
        # 3d [32, 12, 3, 3, 3]
        # useful? https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=number_of_modules, kernel_size=(1, 3), stride=(1, 3)).to(self.device)

        # self.conv1 = nn.Conv2d(in_channels=3*(number_of_modules + 1), out_channels=32, kernel_size=(1, 1)).to(self.device)
        # self.bn1 = nn.BatchNorm1d(32, affine=False).to(self.device)

        # ceil mode needs to be true or else a 0 column is added to both sides
        self.p1 = nn.AvgPool1d(3, ceil_mode=False)
        # self.bn1 = nn.BatchNorm1d(32, affine=False).to(self.device)
        self.bn1 = nn.BatchNorm1d(4*number_of_modules, affine=False).to(self.device)

        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=(3,1)).to(self.device)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=number_of_modules+1, kernel_size=(1, 3), stride=(1, 3)).to(self.device)
        self.p2 = nn.AvgPool1d(3, ceil_mode=False)
        # self.bn2 = nn.BatchNorm1d(64, affine=False).to(self.device)
        self.bn2 = nn.BatchNorm1d(4*number_of_modules, affine=False).to(self.device)

        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(3,1)).to(self.device)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=number_of_modules+1, kernel_size=(1, number_of_modules+1)).to(self.device)
        self.p3 = nn.AvgPool1d(3)
        # self.bn3 = nn.BatchNorm1d(128, affine=False).to(self.device)
        self.bn3 = nn.BatchNorm1d(4*number_of_modules, affine=False).to(self.device)

        # Input for the action input
        # x = torch.rand((32, 3*(number_of_modules + 1) + (number_of_modules))).to(self.device).view(32, 3*(number_of_modules + 1) + (number_of_modules), 1, 1)

        # x = torch.rand((32, 3*(number_of_modules + 1))).to(self.device).view(32, 3*(number_of_modules + 1), 1, 1)
        # x = torch.rand(12, 1, 4*number_of_modules).to(self.device)
        # shape [12, 1, 3]
        # with kernel (1, 3) we should be combining one vector at a time, in bits of 3
        x = torch.tensor([[0., 0., 1., 1., 0., 0., 1., 0., 0.],  # all states
                          [0., 0., 1., 1., 0., 0., 1., 0., 0.],  # all actions
                          [0., 0., 1., 1., 0., 0., 1., 0., 0.],  # all mean action
                          [0., 0., 0., 0., 0., 0., 0., 0., 1.25]  # reward
                          ], dtype=torch.float32).to(self.device).view(12, 1, 1, 3)
        print(f"x >>> {x} (shape) {x.shape}")
        self._to_linear = None
        self.convs(x)

        # self.fc1 = nn.Linear(self._to_linear, 515).to(self.device)
        self.fc1 = nn.Linear(4*number_of_modules, 2*number_of_modules).to(self.device)
        # self.fc2 = nn.Linear(515, 3).to(self.device)
        self.fc2 = nn.Linear(2*number_of_modules, 3).to(self.device)

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
            print(f"conv2 reshape >>> {x.view(4, 1, 4)} (shape) {x.view(4, 1, 4).shape}")
        x = x.view(4, 1, 4)
        # x = self.bn2(x)
        x = self.p2(x)
        if dbg:
            print(f"bn2 >>> {x} (shape) {x.shape}")
        x = F.relu(x)

        if dbg:
            print(f"relu >>> {x} (shape) {x.shape}")
            print(f"relu reshape >>> {x.view(1, 1, 1, 4)} (shape) {x.view(1, 1, 1, 4).shape}")
        # x = self.conv3(x)
        # if dbg:
        #     print(f"conv3 >>> {x} (shape) {x.shape}")
        # x = self.bn3(x)
        # # x = self.p3(x)
        # if dbg:
        #     print(f"bn3 >>> {x}(shape) {x.shape}")
        # x = F.relu(x)
        # if dbg:
        #     print(f"relu >>> {x} (shape) {x.shape}")

        if self._to_linear is None:
            # self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            self._to_linear = x

        return x

    def forward(self, actions, dbg=True):
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
