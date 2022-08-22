import torch
import numpy as np


class Environment():

    def __init__(self, N, action_count, obstacles_map=None):
        """
        Initiates Environment
        :param N: grid size
        :param action_count: number of actions robots can do
        :param obstacles_map: array of obstacles positions
        """

        self.N = N
        self.action_count = action_count

        # Environment matrix
        """
        env[0]  keeping track of obstacles
        env[1]  keeping track of friendly locations
        env[2]  keeping track of ALL previous locations
        env[3]  keeping track of personal CURRENT location
        env[4]  keeping track of ALL OF THE ROBOT POS 
        """
        self.env = torch.zeros((5, N, N))

        self.done = False   # True if robots are done with map
        self.p_completion = 0.0     # percent of covered territory

        self.obstacles_map = obstacles_map
        self.set_obstacles(self.obstacles_map)

    # def actions(self, action, agent):

    def set_obstacles(self, obstacles=None):
        # TODO: replace manual with upload
        if not obstacles:
            self.env[0][7][6] = 1
            self.env[0][7][7] = 1
            self.env[0][7][8] = 1
            self.env[0][8][6] = 1
            self.env[0][8][7] = 1
            self.env[0][8][8] = 1

    # Setting the last matrix to agent id so that we can easily extract
    #   friendly layers and personal layer when needed
    def set_personal_pos(self, agent=None, x=0, y=0, reset=None):
        if reset:
            self.env[4][x][y] = agent.agent_id
        else:
            self.env[4][agent.previous_x][agent.previous_y] = 0
        self.env[4][x][y] = agent.agent_id

    # resetting map
    def reset(self, agent=None, x=0, y=0):
        self.env = torch.zeros((5, self.N, self.N))
        self.p_completion = 0.0
        self.set_obstacles(self.obstacles_map)

        if agent:
            agent.steps = 0
            agent.stay_count = 0
            agent.x_coordinate = x
            agent.y_coordinate = y
            agent.previous_x = x
            agent.previous_y = y
            self.set_personal_pos(agent, x, y)

        return self.env

    # getting (4,N,N) state out of the (5,N,N) state,
    #   with modified Friendlies and Personal layers,
    #   omitting the last (ALL) layer
    def get_state(self, agent):
        # clone of env
        agent_env = self.env
        agent_pos = np.where(agent_env[4] == agent.agent_id)
        # marking robot's current position
        agent_env[3][agent_pos[0][0]][agent_pos[1][0]] = 1
        # copying current all robot positions to layer 1
        agent_env[1] = agent_env[4]
        # setting position of self to 0
        agent_env[1][agent_pos[0][0]][agent_pos[1][0]] = 0
        # getting positions of all the other robots
        pos = np.nonzero(agent_env[1])
        for p in pos:
            # setting position of all the other robots to 1 (instead of their agent_id)
            agent_env[1][p[0]][p[1]] = 1
        # removing matrix with all robot positions  (layer 4)
        agent_env = agent_env[:-1, :, :]
        return agent_env


    #
    def move_agent(self, action, agent):

        # TODO: REWORK THIS TO AVOID COLLISION ACTIONS
        old_x, old_y = agent.x_coordinate, agent.y_coordinate
        actions = {
            0: (agent.x_coordinate, agent.y_coordinate + 1),      # up
            1: (agent.x_coordinate, agent.y_coordinate - 1),      # down
            2: (agent.x_coordinate + 1, agent.y_coordinate),      # right
            3: (agent.x_coordinate - 1, agent.y_coordinate),      # left
            4: (agent.x_coordinate, agent.y_coordinate),          # stay
        }

        """
            This is checking to make sure that the robot is not going out of bounds    
        """
        new_state = actions[action.flatten()[0].item()]
        agent.x_coordinate = new_state[0]
        agent.y_coordinate = new_state[1]

        if agent.x_coordinate > self.N:
            agent.x_coordinate = old_x
        elif agent.x_coordinate < 0:
            agent.x_coordinate = old_x
        if agent.y_coordinate > self.N:
            agent.y_coordinate = old_y
        elif agent.y_coordinate < 0:
            agent.y_coordinate = old_y

        # checking if agent is hitting an obstacle or another agent
        if self.env[0][agent.x_coordinate][agent.y_coordinate] == 1 \
                or self.env[4][agent.x_coordinate][agent.y_coordinate] != 0:
            agent.x_coordinate = old_x
            agent.y_coordinate = old_y
            return old_x, old_y

        return old_x, old_y

    # TODO: remove and make single function
    @staticmethod
    def get_action(action, x_coordinate, y_coordinate):
        actions = {
            0: (x_coordinate, y_coordinate + 1),    # UP
            1: (x_coordinate, y_coordinate - 1),    # DOWN
            2: (x_coordinate + 1, y_coordinate),    # RIGHT
            3: (x_coordinate - 1, y_coordinate),    # LEFT
            4: (x_coordinate, y_coordinate)         # STAY
        }
        return actions[action]

    def check_action(self, action, x_coordinate, y_coordinate):
        action_space = []
        action = action.flatten().tolist()
        for i in range(len(action)):
            state = self.get_action(i, x_coordinate, y_coordinate)
            if state[0] > self.N - 1:
                action_space.append(float("-inf"))
                continue
            if state[0] < 0:
                action_space.append(float("-inf"))
                continue
            if state[1] > self.N - 1:
                action_space.append(float("-inf"))
                continue
            if state[1] < 0:
                action_space.append(float("-inf"))
                continue
            action_space.append(action[i])
        return torch.tensor(action_space).view(1, 1, -1)

    def step(self, action, agent):
        # TODO: rework
        done = False
        reward = 0.0

        old_x, old_y = self.move_agent(action, agent)

        # New Lazy Penalty
        if agent.previous_x == agent.x_coordinate and agent.previous_y == agent.y_coordinate:
            agent.stay_count += 1
            reward -= 0.1 * agent.stay_count
        else:
            agent.stay_count = 0

        if self.env[2][agent.x_coordinate][agent.y_coordinate] == 0: #0 = unvisited state
            reward += 2
        elif self.env[2][agent.x_coordinate][agent.y_coordinate] == 1: #2 visited state
            reward -= 0.5

        # replacing 1 from previous state
        self.env[1][old_x][old_y] = 0
        self.env[4][old_x][old_y] = 0
        # 1 = current state
        self.env[1][agent.x_coordinate][agent.y_coordinate] = 1
        self.env[4][agent.x_coordinate][agent.y_coordinate] = 1
        # visited state marking
        self.env[2][old_x][old_y] = 1
        # the current location should be taken into account while calculating the done flag/completion %
        # TODO: double check
        self.p_completion = (len(torch.nonzero(torch.tensor(self.env[2]))) +
                             len(torch.nonzero(torch.tensor(self.env[0])))) / (self.N * self.N)

        if (len(torch.nonzero(torch.tensor(self.env[2]))) +
            len(torch.nonzero(torch.tensor(self.env[0])))) == self.N * self.N:
            done = True
            reward += 200

        agent_env = self.get_state(agent)

        return agent_env, old_x, reward, done, old_y
