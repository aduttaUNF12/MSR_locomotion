import struct
import math
import sys
import itertools

import numpy as np

from .constants import EPSILON, NUM_MODULES, GAMMA, MIN_EPSILON, MEMORY_CAPACITY, T, BATCH_SIZE, LEADER_ID
from .agent import Agent
from .buffers import ReplayBuffer
from .loggers import writer

from controller import Supervisor


class Action:
    def __init__(self, name, function):
        self.name = name
        self.func = function


# robot module instance
class Module(Supervisor):
    def __init__(self, target_net, policy_net, nn_type):
        Supervisor.__init__(self)
        #  webots section
        self.bot_id = int(self.getName()[7])
        self.timeStep = int(self.getBasicTimeStep())
        self.prev_episode = 1
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_mean_action = []
        self.prev_episode_mean_action = []
        self.episode_current_action = []
        # self.module = module
        self.nn_type = nn_type
        self.episode = 1
        self.total_time_elapsed = 0
        self.self_message = bytes

        self.gps = self.getDevice("gps")
        self.gps.resolution = 0
        self.gps.enable(self.timeStep)

        self.emitter = self.getDevice("emitter")

        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.timeStep)

        self.motor = self.getDevice("motor")

        self.receiver.setChannel(5)

        self.emitter.setChannel(5)

        self.actions = [
            Action("Up", self.action_up),
            Action("Down", self.action_down),
            Action("Neutral", self.action_neutral)
        ]

        self.old_pos = self.gps.getValues()
        self.new_pos = self.old_pos
        self.t = 0
        self.batch_ticks = 1
        self.reward = 0
        self.loss = None
        self.calc_reward()

        self.global_actions = [1]*NUM_MODULES
        self.global_states = [1]*NUM_MODULES
        self.global_states_vectors = []
        self.global_actions_vectors = []
        self.global_prev_actions_vectors = []
        self.mean_action_vector = []
        self.mean_action = 0
        self.prev_states_vector = []
        self.prev_mean_action_vector = []
        self.prev_mean_action = 0
        """
        states:  0 - min
                 1 - zero
                 2 - max
        """
        # TODO: it is a collection of all of the module states, but later will have to be only for neighbors
        self.current_state = 1
        self.prev_state = self.current_state
        self.act = None
        self.sts = None


        # getting leader to decide initial actions
        self.current_action = np.random.choice([i for i in range(3)], 1)[0]
        self.prev_actions = self.current_action

        # TEMP FOR ERROR CHECKING TODO
        self.tries = 0
        self.batch_updated = False
        self.re_adjust = False

        self.min_max_set = False
        self.min_batch = 0
        self.max_batch = 0
        self.recycles = 0

        # BUFFERS
        self.REPLAY_MEMORY_EPISODE = 1
        self.ReplayMemory_EpisodeBuffer = {}

        self.replay_buf_reward = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
        self.replay_buf_action = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.int32)
        self.replay_buf_state = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
        self.replay_buf_state_ = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
        self.replay_buf_mean_action = ReplayBuffer(shape=(MEMORY_CAPACITY,), dtype=np.float64)
        #  TODO: possibly rework these \/
        self.LAST_MEAN_ACTION_INDEX = 0
        self.LAST_STATE_INDEX = 0
        self.LAST_ACTION_INDEX = 0
        self.Replay_Buf_Vector_States = []
        self.Replay_Buf_Vector_States_ = []
        self.Replay_Buf_Vector_Mean_Actions = []
        self.Replay_Buf_Vector_Mean_Actions_ = []
        self.Replay_Buf_Vector_Actions = []
        self.Replay_Buf_Vector_Actions_ = []

        # making all modules have NN
        self.agent = Agent(self.bot_id, NUM_MODULES, 3, 0.001, gamma=GAMMA,
                           epsilon=EPSILON, eps_dec=0.001, eps_min=MIN_EPSILON,
                           target_net=target_net, policy_net=policy_net, nn_type=nn_type)

        self.buffer_overflow = 0
        self.state_changer()
        self.notify_of_an_action(self.current_action)
        # self.notify_of_an_action(self.current_state, action=False)

    # notify other modules of the chosen action
    # if action is True means that an action is being transmitted, if False current state is being transmitted
    def notify_of_an_action(self, state, action=True):
        message = struct.pack("i", self.bot_id)
        if action:
            # 1 if sending action
            self.global_actions[self.bot_id-1] = state
            message += struct.pack("i", 0)
        else:
            # 2 if sending state
            self.global_states[self.bot_id-1] = state
            message += struct.pack("i", 1)

        message += struct.pack("i", state)

        ret = self.emitter.send(message)

        if ret == 1:
            pass
        elif ret == 0:
            print(f"[*] {self.bot_id} - acts were not sent")
            exit(3)
        else:
            print(f"[*] {self.bot_id} - error while sending data")
            exit(123)

        # self.self_message = message

    # gets actions which were sent by the leader
    # if action is True means that an action is being received, if False current state is being received
    def get_other_module_actions(self):
        if self.receiver.getQueueLength() > 0:
            while self.receiver.getQueueLength() > 0:
                message = self.receiver.getData()
                # data = [bot id, action/state flag, payload]
                data = struct.unpack("i" + "i" + "i", message)
                # print(f"[{self.bot_id}] got emitter packet: {data}", file=sys.stderr)

                if data[1] == 0:
                    self.global_actions[data[0]-1] = data[2]
                else:
                    self.global_states[data[0]-1] = data[2]

                self.receiver.nextPacket()
        else:
            # print(f"[{self.bot_id}] Error: no packets in receiver queue.")
            return 2

    def action_up(self):
        self.motor.setPosition(1.57)

    def action_down(self):
        self.motor.setPosition(-1.57)

    # no action is taken, module just passes
    def action_neutral(self):
        # TODO: REMOVE IF HAS ODD RESULTS
        # experiment to see if making neutral not do anything will be beneficial or not
        # self.motor.setPosition(0)
        pass

    # calculates reward based on the travel distance from old position to the new, calculates hypotenuse of two points
    def calc_reward(self):
        self.reward = math.hypot(self.old_pos[0]-self.new_pos[0], self.old_pos[2]-self.new_pos[2])
        # logger(bot_id=self.bot_id, reward=self.reward)

    # changes the state of the module based on the action.
    #   if the current state is down (0) and action is 0 (up) the state will be neutral (1)
    #   same logic backwards, is state is neutral (1) and action is 1 (down), the state will be down(0)
    #   if state is already up/down and we get action up/down module stays in current position
    def state_changer(self):
        self.prev_state = self.current_state

        if int(self.current_action) == 0:
            if int(self.current_state) < 2:
                self.current_state += 1

        elif int(self.current_action) == 1:
            if int(self.current_state) > 0:
                self.current_state -= 1

        elif int(self.current_action) == 2:
            pass

        else:
            print(f"[{self.bot_id}]  Error with state_changer !!!!!!!!!!", file=sys.stderr)
            exit(2)

    # Converts global_actions (actions of all of the modules in the chain) into a vector representation
    def actions_to_vectors(self):
        # TODO: make a log of prev vectors and current vectors to see if everything works
        if self.re_adjust:
            self.Replay_Buf_Vector_Actions_[self.LAST_STATE_INDEX] = self.global_prev_actions_vectors[0:NUM_MODULES] \
                if len(self.global_prev_actions_vectors) >= NUM_MODULES else [[0, 0, ]*NUM_MODULES]
        else:
            self.Replay_Buf_Vector_Actions_.append(self.global_prev_actions_vectors[0:NUM_MODULES]
                                                   if len(self.global_prev_actions_vectors) >= NUM_MODULES
                                                   else [[0, 0, 0]]*NUM_MODULES)
        self.global_actions_vectors = []
    # adds vector representation of the state to an array, [x, y, z]
        for action in self.global_actions:
            if int(action) == 1:
                self.global_actions_vectors.append([0, 0, 0])
            elif int(action) == 0:
                self.global_actions_vectors.append([0, -1, 0])
            elif int(action) == 2:
                self.global_actions_vectors.append([0, 1, 0])
            else:
                print(f"[{self.bot_id}] Error with state_to_vectors !!!!!!!!!!!!!", file=sys.stderr)
                exit(2)

        if self.re_adjust:
            self.Replay_Buf_Vector_Actions[self.LAST_ACTION_INDEX] = self.global_actions_vectors[0:NUM_MODULES]
            self.LAST_ACTION_INDEX += 1
        else:
            self.Replay_Buf_Vector_Actions.append(self.global_actions_vectors[0:NUM_MODULES])

        # self.global_prev_actions_vectors = self.global_actions_vectors
        self.calc_mean_action_vector()
        # self.global_actions_vectors = []

    # Converts global states to vector form
    def states_to_vectors(self):
        """
       states:  0 - min
                1 - zero
                2 - max
       """
        # adds vector representation of the state to an array, [Down, Neutral, Up]
        # reasoning for line 517 is calc_mean_action_vector()
        if self.re_adjust:
            self.Replay_Buf_Vector_States_[self.LAST_STATE_INDEX] = self.prev_states_vector[0:NUM_MODULES] \
                if len(self.prev_states_vector) >= NUM_MODULES else [[0, 0, ]*NUM_MODULES]
        else:
            self.Replay_Buf_Vector_States_.append(self.prev_states_vector[0:NUM_MODULES]
                                                  if len(self.prev_states_vector) >= NUM_MODULES
                                                  else [[0, 0, 0]]*NUM_MODULES)

        self.global_states_vectors = []
        for state in self.global_states:
            if int(state) == 1:
                self.global_states_vectors.append([0, 0, 0])
            elif int(state) == 0:
                self.global_states_vectors.append([1, 0, 0])
            elif int(state) == 2:
                self.global_states_vectors.append([0, 0, 1])
            else:
                print(f"[{self.bot_id}] Error with state_to_vectors !!!!!!!!!!!!!", file=sys.stderr)
                exit(2)

        # for some reason there is an extra [0] array at the end
        if self.re_adjust:
            self.Replay_Buf_Vector_States[self.LAST_STATE_INDEX] = self.global_states_vectors[0:NUM_MODULES]
            self.LAST_STATE_INDEX += 1
        else:
            self.Replay_Buf_Vector_States.append(self.global_states_vectors[0:NUM_MODULES])

    # Loops through NUM_MODULES+1 (since there is no module 0), sums actions of all modules except for current
    # divides by the total number of modules
    def calc_mean_action(self):
        a = 0
        for m in range(NUM_MODULES):
            # self.bot_id - 1 since bot ids start with 1 while arrays with index of 0
            if m != (self.bot_id - 1):
                try:
                    a += self.global_actions[m]
                except IndexError:
                    pass
        # NUM_MODULE - 1 since we are taking a mean of all except current modules
        self.prev_mean_action = self.mean_action
        self.mean_action = a / (NUM_MODULES - 1)
        # logger(mean_action=self.mean_action, a=a)
        # self.mean_action_rounder()

    # Calculates the mean of the action vectors
    def calc_mean_action_vector(self):
        # Making a base vector the shape of (1, NUM_MODULES)
        a = [0]*NUM_MODULES
        # add previous mean action vector to the buffer
        #   if previous mean action vector has size greater than/equal to number of modules, this means that either
        #   previous mean action vector has more values than actions so limit the input to only the first NUM_MODULES
        #   actions; else the previous mean action vector is just being initialized so fill with 0 vectors
        if self.re_adjust:
            self.Replay_Buf_Vector_Mean_Actions_[self.LAST_MEAN_ACTION_INDEX] = self.prev_mean_action_vector[0:NUM_MODULES] \
                if len(self.prev_mean_action_vector) >= NUM_MODULES else [0]*NUM_MODULES
        else:
            self.Replay_Buf_Vector_Mean_Actions_.append(self.prev_mean_action_vector[0:NUM_MODULES]
                                                        if len(self.prev_mean_action_vector) >= NUM_MODULES
                                                        else [0]*NUM_MODULES)
        for m in range(NUM_MODULES):
            # self.bot_id - 1 since bot ids start with 1 while arrays with index of 0
            if m != (self.bot_id - 1):
                try:
                    a = np.add(a, self.global_actions_vectors[m])
                except IndexError:
                    pass
        self.prev_mean_action_vector = self.mean_action_vector
        # NUM_MODULE - 1 since we are taking a mean of all except current modules
        self.mean_action_vector = np.true_divide(a, (NUM_MODULES - 1))

        if self.re_adjust:
            self.Replay_Buf_Vector_Mean_Actions[self.LAST_MEAN_ACTION_INDEX] = self.mean_action_vector[0:NUM_MODULES]
            self.LAST_MEAN_ACTION_INDEX += 1
        else:
            self.Replay_Buf_Vector_Mean_Actions.append(self.mean_action_vector[0:NUM_MODULES])

    # Takes mean_action and rounds it down
    def mean_action_rounder(self):
        dec, n = math.modf(float(self.mean_action))
        if dec < 0.5:
            n = n
        else:
            n = n + 1
        self.mean_action = n

    def learn(self):
        # If current action is None
        if self.act is None:
            # get actions which were send by other modules, if ret is 2 that means that not all modules sent their
            # actions yet, so loop while you wait for them to come
            ret = self.get_other_module_actions()
            if ret == 2:
                if self.tries > 100:
                    print("Error with tries")
                    exit(11)
                self.tries += 1
                return
            # print(f"[{self.bot_id}] got all of the actions: {self.global_actions}", file=sys.stderr)
            self.tries = 0
            # select an action corresponding to the current_action number
            self.act = self.actions[self.current_action]
            # self.calc_mean_action()
            self.actions_to_vectors()
            self.calc_mean_action_vector()

            self.notify_of_an_action(self.current_state, action=False)

        if self.sts is None:
            # print(f"[{self.bot_id}] inside self.sts None", file=sys.stderr)

            ret = self.get_other_module_actions()
            if ret == 2:
                if self.tries > 100:
                    print("Error with tries")
                    exit(11)
                self.tries += 1
                return

            # print(f"[{self.bot_id}] got all of the states: {self.global_states}", file=sys.stderr)
            self.tries = 0
            self.sts = 1
            self.states_to_vectors()

        elif self.t + (self.timeStep / 1000.0) < T:
            # carry out a function
            self.act.func()

        elif self.t < T:
            # get new position
            self.new_pos = self.gps.getValues()
            # calculate the reward
            self.calc_reward()
            # if reward is missing, skip. There were errors when Webots would not properly assign GPS coordinates
            # causing reward to be none
            if self.reward != self.reward or self.reward is None:
                pass
            else:
                if not self.min_max_set:
                    self.min_batch = self.replay_buf_state.return_buffer_len
                    self.min_max_set = True

                self.replay_buf_state.return_buffer_len += 1
                if self.replay_buf_state.return_buffer_len > MEMORY_CAPACITY:
                    print(f"Replay buffer is full")
                    self.replay_buf_state.return_buffer_len = 0
                    self.replay_buf_state.clear()
                    self.replay_buf_state_.clear()
                    self.replay_buf_mean_action.clear()
                    self.replay_buf_action.clear()
                    self.replay_buf_reward.clear()
                    self.buffer_overflow = 1
                    # exit(11)

                self.episode_reward += self.reward
                self.episode_rewards.append(self.reward)
                self.episode_mean_action.append(self.mean_action)
                self.prev_episode_mean_action.append(self.prev_mean_action)

                self.episode_current_action.append(self.current_action)
                # self.episode_current_action.append(self.global_actions)

                # If Episode changed
                if self.episode > self.prev_episode:
                    self.max_batch = self.replay_buf_state.return_buffer_len
                    self.min_max_set = False
                    if self.buffer_overflow != 1:
                        self.ReplayMemory_EpisodeBuffer[self.episode-1] = {"min": self.min_batch,
                                                                           "max": self.max_batch}

                        # Since Episode 1 usually is 1 less than all other episodes
                        if self.episode - 1 == 1:
                            self.ReplayMemory_EpisodeBuffer[self.REPLAY_MEMORY_EPISODE] = {"min": self.min_batch,
                                                                                      "max": self.max_batch + 1}
                        else:
                            self.ReplayMemory_EpisodeBuffer[self.REPLAY_MEMORY_EPISODE] = {"min": self.min_batch,
                                                                                  "max": self.max_batch}
                    else:
                        self.buffer_overflow = 0

                    self.replay_buf_reward.put(np.array(self.episode_rewards))
                    self.replay_buf_state.put(np.array(self.current_state))
                    self.replay_buf_mean_action.put(np.array(self.episode_mean_action))
                    self.replay_buf_state_.put(np.array(self.prev_state))
                    self.replay_buf_action.put(np.array(self.episode_current_action))

                    #
                    # if REPLAY_MEMORY_EPISODE == EPISODE_LIMIT:
                    #     # .clear() resets buffer index to 0
                    #     # this part just resets all indexes to initial index
                    #     replay_buf_action.clear()
                    #     replay_buf_state_.clear()
                    #     replay_buf_reward.clear()
                    #     replay_buf_state.clear()
                    #     replay_buf_mean_action.clear()
                    #     REPLAY_MEMORY_EPISODE = 0
                    #     self.min_batch = 0
                    #     self.max_batch = 0
                    #     replay_buf_state.return_buffer_len = 0
                    #     LAST_STATE_INDEX = 0
                    #     LAST_MEAN_ACTION_INDEX = 0
                    #     # for debugging, will be removed later
                    #     self.recycles += 1

                    if self.bot_id == LEADER_ID:
                        # logger
                        writer(self.bot_id, NUM_MODULES, self.total_time_elapsed,
                               self.episode_reward, self.loss, self.episode, self.nn_type)

                    # if EPISODE > 3 and not self.re_adjust:
                    #     # After 3 episodes amount of elements in each episode becomes more obvious and stable
                    #     #   so all of the buffers are switching to structure which will work with
                    #     #   circular buffer structure
                    #     self.re_adjust = True
                    #     # Vector Actions x6 Vector states x3
                    #     # size of actions is x6 (for some reason mean actions take double the size of states)
                    #     #   x Episode_Limit to make sure that all parts can fit, + 1000 as a buffer safety area
                    #     size_of_mean_actions = (len(self.episode_mean_action)*6)*EPISODE_LIMIT + 1000
                    #     size_of_states = (len(self.episode_current_action)*3)*EPISODE_LIMIT + 1000
                    #
                    #     temp_actions = Replay_Buf_Vector_Mean_Actions
                    #     Replay_Buf_Vector_Mean_Actions = [None]*size_of_mean_actions
                    #
                    #     for index, item in enumerate(temp_actions):
                    #         Replay_Buf_Vector_Mean_Actions[index] = item
                    #
                    #     temp_actions = Replay_Buf_Vector_Mean_Actions_
                    #     Replay_Buf_Vector_Mean_Actions_ = [None]*size_of_mean_actions
                    #
                    #     for index, item in enumerate(temp_actions):
                    #         Replay_Buf_Vector_Mean_Actions_[index] = item
                    #
                    #     # last accessible mean action is LAST_MEAN_ACTION_INDEX - 1
                    #     LAST_MEAN_ACTION_INDEX = len(temp_actions)
                    #     del temp_actions
                    #
                    #     temp_states = Replay_Buf_Vector_States
                    #     Replay_Buf_Vector_States = [None]*size_of_states
                    #
                    #     for index, item in enumerate(temp_states):
                    #         Replay_Buf_Vector_States[index] = item
                    #
                    #     temp_states = Replay_Buf_Vector_States_
                    #     Replay_Buf_Vector_States_ = [None]*size_of_states
                    #
                    #     for index, item in enumerate(temp_states):
                    #         Replay_Buf_Vector_States_[index] = item
                    #
                    #     # last accessible mean action is LAST_STATE_INDEX - 1
                    #     LAST_STATE_INDEX = len(temp_states)
                    #     del temp_states, index, item

                    self.episode_reward = 0
                    self.episode_mean_action.clear()
                    self.prev_episode_mean_action.clear()
                    self.episode_current_action.clear()
                    self.prev_episode = self.episode
                    self.batch_ticks += 1
                    self.batch_updated = False
                    self.REPLAY_MEMORY_EPISODE += 1

                # batch is at least at the minimal working size
                # if self.batch_ticks >= BATCH_SIZE and not self.batch_updated:
                if self.batch_ticks >= BATCH_SIZE:
                    # run the NN and collect loss
                    self.loss = self.agent.optimize(
                        batch=True, episode_buffer=self.ReplayMemory_EpisodeBuffer,
                        action_buffer=self.replay_buf_action,
                        reward_buffer=self.replay_buf_reward,
                        vector_states_buffer=self.Replay_Buf_Vector_States,
                        vector_states__buffer=self.Replay_Buf_Vector_States_,
                        vector_mactions_buffer=self.Replay_Buf_Vector_Mean_Actions,
                        vector_mactions__buffer=self.Replay_Buf_Vector_Mean_Actions_,
                        vector_actions_buffer=self.Replay_Buf_Vector_Actions,
                        vector_actions__buffer=self.Replay_Buf_Vector_Actions,
                        episode=self.episode
                                                    )
                    # if self.bot_id == LEADER_ID:
                    #     print(f"loss: {self.loss}")
                    self.batch_updated = True
                    self.batch_ticks = 0

            # set previous action option to the new one
            self.prev_actions = self.current_action
            # get the array of state vectors
            robot_state_vectors = self.global_states_vectors[0:NUM_MODULES]
            # 540-541 is for input with action
            # t = self.global_actions_vectors[self.bot_id-1]
            # robot_state_vectors.append(t)
            # robot_state_vectors.append(self.global_actions_vectors)
            robot_state_vectors.append(list(itertools.chain(*self.global_actions_vectors)))
            robot_state_vectors.append(self.mean_action_vector)
            robot_state_vectors.append([self.reward])

            self.current_action = self.agent.choose_action(robot_state_vectors)[0]
            self.prev_states_vector = self.global_states_vectors
            self.global_states_vectors = []
            # run the action and change the state
            self.state_changer()
            # notify others of current action
            self.notify_of_an_action(self.current_action)

        else:
            self.act = None
            self.sts = None
            self.old_pos = self.gps.getValues()
            self.t = 0

