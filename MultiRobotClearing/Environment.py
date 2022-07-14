import math
import random
import sys

import torch
import numpy as np

class Environment():
    """
            position [0] is keeping track of the obstacle locations
            position [1] is keeping track of the robots' location
            position [2] is keeping track of all the previous positions the robot has visited
    """
    def __init__(self, N):
        self.N = N
        self.environment = torch.zeros((3,self.N, self.N))
        self.action_count = 9
        self.states = self.N * self.N * 3
        self.done = False
        self.p_completion = 0.0
        self.environment[0][7][6] = 1
        self.environment[0][7][7] = 1
        self.environment[0][7][8] = 1
        self.environment[0][8][6] = 1
        self.environment[0][8][7] = 1
        self.environment[0][8][8] = 1

        self.visit_count = {}
        
    

    """
    This will reset the environment and set the agent to position (1, 1) in the grid
    
    """

#
    def init_reset(self):
        self.environment = torch.zeros((3 ,16, 16))
        self.p_completion = 0.0

    def reset(self, agent, x=0, y=0):
        # self.agent.model_count = 0
        # self.agent.random_count = 0
        agent.model_count = 0
        agent.random_count = 0

        # self.agent.steps = 0
        # self.agent.steps_given = self.budget
        agent.steps = 0
        # self.environment[1][self.agent.x_coordinate][self.agent.y_coordinate] = self.budget + 10
        # self.environment[1][agent.x_coordinate][agent.y_coordinate] = 1
        self.environment[1][x][y] = 1
        # self.agent.x_coordinate = x
        # self.agent.y_coordinate = y
        agent.x_coordinate = x
        agent.y_coordinate = y
        # Duplicate code??
        # self.environment[1][self.agent.x_coordinate][self.agent.y_coordinate] = self.budget + 10
        self.environment[1][x][y] = 1
        self.environment[0][7][6] = 1
        self.environment[0][7][7] = 1
        self.environment[0][7][8] = 1
        self.environment[0][8][6] = 1
        self.environment[0][8][7] = 1
        self.environment[0][8][8] = 1

        
        return self.environment

    
    #returns the state of which the robot is in
    def get_state(self):
        return self.environment

    def step(self, action, agent, agent_id):
        done = False
        reward = 0
        
        """
            How close the robot is to the nearest charging station, as it is approaching the charging station it gets
            an incremental reward

            - after each step keep track of the nearest charging station 

            - a function that can calculate the distance to the nearest charging station
            
        """
        
        old_x, old_y, hit = self.move_robot(action, agent)

        # keeping track of the amount of times robot visited this state
        if self.visit_count.get(agent_id) is None:
            self.visit_count[agent_id] = {old_x: {old_y: 1}}
        if self.visit_count[agent_id].get(old_x) is None:
            self.visit_count[agent_id][old_x] = {old_y: 1}
        else:
            if self.visit_count[agent_id][old_x].get(old_y) is None:
                self.visit_count[agent_id][old_x][old_y] = 1
            else:
                self.visit_count[agent_id][old_x][old_y] += 1

        # self.agent.steps_given -= 1
        # agent.steps_given -= 1

        if self.environment[2][agent.x_coordinate][agent.y_coordinate] == 0: #0 = unvisited state
            reward += 1
        elif self.environment[2][agent.x_coordinate][agent.y_coordinate] == 1: #2 visited state
            # LAZY PENALTY
            if action == 8:
                reward -= 0.1
            else:
                try:
                    # reward += -1*self.visit_count[agent_id][agent.x_coordinate][agent.y_coordinate]
                    reward += -1
                except KeyError:
                    reward += -1

        # HIT AN OBSTACLE
        if hit:
            reward -= 0.5
        """
        each state is represented as [0, 0, 0]cla

        for each index in the state will be described below:
            position [0] is keeping track of of the obstacle locations
            position [1] is keeping track of the robots location
            position [2] is keeping track of all the previous positions the robot has visited
        Line 157: is setting the previous state back to 0
        Line 158: is setting the current position of the robot
        Line 159: is setting the value of the third index to two to keep track of the robots steps
        """
        self.environment[1][old_x][old_y] = 0   # replacing 1 from previous state
        # self.environment[1][self.agent.x_coordinate][self.agent.y_coordinate]= self.agent.steps_given + 10 #1 = current state
        self.environment[1][agent.x_coordinate][agent.y_coordinate] = 1 #1 = current state
        self.environment[2][old_x][old_y] = 1   # visited state marking
        # the current location should be taken into account while calculating the done flag/completion %
        self.p_completion = (len(torch.nonzero(torch.tensor(self.environment[2]))) +
                             len(torch.nonzero(torch.tensor(self.environment[0])))) / (self.N * self.N)
        # self.p_completion = len(torch.nonzero(torch.tensor(self.environment[2]))) / (self.N * self.N)
        
        if (len(torch.nonzero(torch.tensor(self.environment[2]))) +
                len(torch.nonzero(torch.tensor(self.environment[0])))) == self.N * self.N:
            done = True
            reward += 200
            
        return self.environment, old_x, reward, done, old_y

    @staticmethod
    def sym_move(action, agent):
        old_x, old_y = agent.x_coordinate, agent.y_coordinate
        actions = {
            0: (agent.x_coordinate, agent.y_coordinate + 1),      #   up
            1: (agent.x_coordinate, agent.y_coordinate - 1),      #   down
            2: (agent.x_coordinate + 1, agent.y_coordinate),      #   right
            3: (agent.x_coordinate - 1, agent.y_coordinate),      #   left
            # 4: (agent.x_coordinate + 1, agent.y_coordinate - 1),  #   diagonal bottom right
            # 5: (agent.x_coordinate - 1, agent.y_coordinate - 1),  #   diagonal bottom left
            # 6: (agent.x_coordinate + 1, agent.y_coordinate + 1),  #   diagonal top right
            # 7: (agent.x_coordinate - 1, agent.y_coordinate + 1),  #   diagonal top left
            4: (agent.x_coordinate, agent.y_coordinate),          #   stay
        }
        """
        This is checking to make sure that the robot is not going out of bounds
        
        """
        new_state = actions[action.flatten()[0].item()]
        agent.x_coordinate = old_x
        agent.y_coordinate = old_y
        return new_state

    def move_robot(self, action, agent):
            old_x, old_y = agent.x_coordinate, agent.y_coordinate
            actions = {
                0: (agent.x_coordinate, agent.y_coordinate + 1),      #   up
                1: (agent.x_coordinate, agent.y_coordinate - 1),      #   down
                2: (agent.x_coordinate + 1, agent.y_coordinate),      #   right
                3: (agent.x_coordinate - 1, agent.y_coordinate),      #   left
                # 4: (agent.x_coordinate + 1, agent.y_coordinate - 1),  #   diagonal bottom right
                # 5: (agent.x_coordinate - 1, agent.y_coordinate - 1),  #   diagonal bottom left
                # 6: (agent.x_coordinate + 1, agent.y_coordinate + 1),  #   diagonal top right
                # 7: (agent.x_coordinate - 1, agent.y_coordinate + 1),  #   diagonal top left
                4: (agent.x_coordinate, agent.y_coordinate),          #   stay
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

            # checking if agent is hitting an obstacle
            if self.environment[0][agent.x_coordinate][agent.y_coordinate] == 1:
                agent.x_coordinate = old_x
                agent.y_coordinate = old_y
                return old_x, old_y, True
            if self.environment[1][agent.x_coordinate][agent.y_coordinate] == 1:
                return old_x, old_y, True
            return old_x, old_y, False

    def get_action(self, action, x_coordinate, y_coordinate):
        actions = {
            0: (x_coordinate, y_coordinate + 1),    # UP
            1: (x_coordinate, y_coordinate - 1),    # DOWN
            2: (x_coordinate + 1, y_coordinate),    # RIGHT
            3: (x_coordinate - 1, y_coordinate),    # LEFT
            # 4: (x_coordinate + 1, y_coordinate - 1),
            # 5: (x_coordinate - 1, y_coordinate - 1),
            # 6: (x_coordinate + 1, y_coordinate + 1),
            # 7: (x_coordinate - 1, y_coordinate + 1),
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

        


