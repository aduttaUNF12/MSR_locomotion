from Environment import Environment
from ExpBuffer import ExpBuffer
from DRQN import Model
from Robot import Robot
from itertools import count, chain
from plotter import Plotter
import numpy as np
import torch
import torch.nn.functional as F
import math
import pandas as pd
import time
#import cv2

import logging

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename="Project2.log", filemode="w", level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 16
NUM_MODULES = 4     # number of robots
robots = []
for r in range(NUM_MODULES):
    robots.append(Robot())
# robot = Robot()
environment = Environment(N)
state_size = 3 * N * N
m_actions = environment.action_count
embedding_size = 8
minibatch = 64
# buffer = 10000 * state_size
buffer = 100000
sample_length = 64
memories = []
main_memory = ExpBuffer(buffer, sample_length)
for robot in robots:
    memories.append(ExpBuffer(buffer, sample_length))
eps_start = 0.9
EPS = eps_start
eps_end = 0.05
# eps_decay = 2000
eps_decay = 600
gamma = 0.99
learning_rate = 0.001
blind_prob = 0
EPISODES = 3000
# EXPLORE = 3000
EXPLORE = EPISODES * 0.2
# EPISODES = 10000


models = []
model_targets = []
adams = []
schedulers = []
for i, r in enumerate(range(NUM_MODULES)):
    models.append(Model(device).double().to(device))
# model = Model().double().to(device)
    model_targets.append(Model(device).double().to(device))
    model_targets[i].load_state_dict(models[i].state_dict())
    model_targets[i].eval()
    adam = torch.optim.Adam(models[i].parameters(), lr=learning_rate)
    adams.append(adam)
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(adam, 'min'))

# =============================================================================
# def convert_to_image(info, i):
#     if i == 0:
#         filename = 'temp.jpg'
#         cv2.imwrite('temp.jpg', info)
#     else:
#         filename = 'temp2.jpg'
#         cv2.imwrite(filename, info)
#     image = cv2.imread(filename)
#     return image
# =============================================================================


def optimize_model(episode, model, model_target, adam, memory, scheduler, r_id):
    EPS = eps_end + (eps_start - eps_end) * \
         math.exp(-1. * (episode - EXPLORE) / eps_decay)
    difference_function = torch.nn.MSELoss()
    # memory = [observations, actions, next_observations, rewards, mean_actions]
    obs = []
    act = []
    n_obs = []
    rwd = []
    macts = []
    for m in memory[0]:
        obs.append(m[r_id])
    for m in memory[1]:
        act.append(m[r_id])
    for m in memory[2]:
        n_obs.append(m[r_id])
    for m in memory[3]:
        rwd.append(m[r_id])
    for m in memory[4]:
        macts.append([m[r_id]])

    obs = torch.cat(obs).to(device)
    act = torch.cat(act).to(device)
    n_obs = torch.cat(n_obs).to(device)
    rwd = torch.tensor(rwd).to(device)
    macts = torch.tensor(macts).to(device)
    obs.requires_grad = True
    n_obs.requires_grad = True

    q_values = model(obs, macts).gather(1, act).to(device)

    target_q_values = model_target(n_obs, macts).max(1)[0].detach().to(device)

    final_results = rwd.double() + (gamma * target_q_values.double())

    adam.zero_grad()


    difference = difference_function(q_values, final_results.unsqueeze(-1).detach())
    difference.backward()

    adam.step()
    scheduler.step(difference)

    return difference, EPS


"""
    pass only one matrix of the current location of iteration 2 

"""
import logging 

epsilon_tracker = []
all_data = []
percentage_covered = []
loss_values = []
episode_coordinates = []
reward_tracker = []
total_steps = 0
start = time.time()

# Initial Environment reset and robot placement
environment.init_reset()
for robot_id, robot in enumerate(robots):
    # print(f"[{robot_id}] ({robot_id*4},{robot_id*4+3})")
    if robot_id == 0:
        environment.reset(x=0, y=0, agent=robot)
    elif robot_id == 1:
        environment.reset(x=0, y=N-1, agent=robot)
    elif robot_id == 2:
        environment.reset(x=N-1, y=0, agent=robot)
    elif robot_id == 3:
        environment.reset(x=N-1, y=N-1, agent=robot)
# print(f"Env1: {environment.environment[1]}")

for episode in range(EPISODES):
    torch.cuda.empty_cache()# for optimization of the code 
    done = False
    hidden = None

    last_action = 0
    ep_rwd = {}
    for robot_id, robot in enumerate(robots):
        ep_rwd[robot_id] = 0.0

    #last_observation = environment.environment
    observation = environment.environment
    coordinates = {}
    for robot_id, robot in enumerate(robots):
        coordinates[robot_id] = []

    step_count = 0
    tracker = 0
    loss_sum = {}
    for robot_id, robot in enumerate(robots):
        loss_sum[robot_id] = 0
    mean_actions = []
    # Initial Environment reset and robot placement
    environment.init_reset()
    for robot_id, robot in enumerate(robots):
        # print(f"[{robot_id}] ({robot_id*4},{robot_id*4+3})")
        if robot_id == 0:
            environment.reset(x=0, y=0, agent=robot)
        elif robot_id == 1:
            environment.reset(x=0, y=N-1, agent=robot)
        elif robot_id == 2:
            environment.reset(x=N-1, y=0, agent=robot)
        elif robot_id == 3:
            environment.reset(x=N-1, y=N-1, agent=robot)
    plot = None
    plot = Plotter(N, environment.environment, environment.environment[0])

    for t in count():
        actions = []
        observations_n = []
        observations = []
        rewards = []
        robot_positions = []
        for robot_id, robot in enumerate(robots):
            if len(mean_actions) > 0:
                m_a = mean_actions[robot_id]
            else:
                m_a = 0.0

            # action, EPS = robot.select_action(torch.tensor(observation).view(1, 3, N, N), models[robot_id], environment, EPS, mean_action=torch.tensor(m_a).view(1,1))

            try:
                while True:
                    action, EPS = robot.select_action(torch.tensor(observation).view(1, 3, N, N), models[robot_id], environment, EPS, mean_action=torch.tensor(m_a).view(1,1))
                    if len(robot_positions) > 0:
                        cur_state = environment.sym_move(action, robot)
                        status = True
                        for pos in robot_positions:
                            if cur_state[0] == pos[0] and cur_state[1] == pos[1]:
                                status = True
                            else:
                                status = False
                        if not status:
                            break
                    else:
                        break
            except RuntimeError as e:
                print(e)
                print("runtime ERROR")
                print("ma")
                print(m_a)
                print("mean actions")
                print(mean_actions)
                print(len(mean_actions))
                exit(1)
            # print(f"[{robot_id}] action taken: {action[0][0]}  Episode: {episode}")
            sym = environment.sym_move(action, robot)
            robot_positions.append(sym)
            plot.move(robot_id+1, robot.x_coordinate, robot.y_coordinate)
            next_observation, old_x, reward, done, old_y = environment.step(action, robot, robot_id)
            # plot.move(robot_id+1, robot.x_coordinate, robot.y_coordinate)
            observations_n.append(next_observation.view(1, 3, N, N))
            actions.append(action)
            observations.append(observation.view(1, 3, N, N))
            rewards.append(reward)
            total_steps += 1
            #if environment.agent.steps_given >= 0:
            reward = torch.tensor([reward])
            observation = next_observation
            ep_rwd[robot_id] += reward    # keep track of episodic reward
            coordinates[robot_id].append((robot.x_coordinate, robot.y_coordinate))
            step_count += 1
            # print(f"next observation: {next_observation}\nreward: {reward}\ndone: {done}")
        # print(f"completion: {environment.p_completion}(EP: {episode})\nMap: {environment.environment[2]}\nLocations: {environment.environment[1]}")

        mean_actions.clear()
        for i, d in enumerate(actions):
            temp = 0.0
            for ii, di in enumerate(actions):
                if i != ii:
                    temp += actions[ii]
            temp /= len(actions)-1
            mean_actions.append(torch.tensor([temp]).view(1,1))
        plot.graph(episode, t)
        # exit(1)
        main_memory.push(observations, actions, observations_n, rewards, mean_actions)
        # if t == 3 * N * N:
        if t == N * N:
            done = True
        if done:
            percentage_covered.append((episode, environment.p_completion))
            episode_coordinates.append((episode, coordinates))
            epsilon_tracker.append((episode, EPS))
            ep_rwd_avg = 0.0
            for robot_id, robot in enumerate(robots):
                ep_rwd_avg += ep_rwd[robot_id]
            ep_rwd_avg = ep_rwd_avg/len(robots)
            # reward_tracker.append((episode, int(ep_rwd)))
            reward_tracker.append((episode, float(ep_rwd_avg)))
            all_data.append((episode, environment.p_completion, EPS, int(ep_rwd_avg), t, loss_sum[0]))
            end = time.time()
            # print(f"Comp. {environment.p_completion*100: .2f}%, ep.: {episode} st: {t}, EPS: {EPS: .2f}, T: {(end-start)/60: .2f} mins., R: {int(ep_rwd)}")
            print(f"Comp. {environment.p_completion*100: .2f}%, ep.: {episode} st: {t}, EPS: {EPS: .2f}, T: {(end-start)/60: .2f} mins., R: {float(ep_rwd_avg)}")
            print(f"Total number of steps were {t}")
            plot.to_gif(episode)
            with open("./graphs/Episode_{}/reward_{}_{}.txt".format(episode, "p" if ep_rwd_avg > 0 else "n", float(ep_rwd_avg)), "w") as fout:
                fout.write("{}\n".format(float(ep_rwd_avg)))
            break
        if episode > EXPLORE:
            observations, actions, next_observations, rewards, mean_actions_ = main_memory.sample_with_batch(minibatch)
            # possible indexing of named tuples
            push_data = [observations, actions, next_observations, rewards, mean_actions_]
            for robot_id, robot in enumerate(robots):
                loss, eps = optimize_model(episode, models[robot_id], model_targets[robot_id],
                                           adams[robot_id], push_data, schedulers[robot_id], robot_id)
                EPS = eps
                loss_sum[robot_id] += float(loss) #float(loss) is for the optimization of the code.
                tracker = t
    if episode > EXPLORE:
        loss_avg = 0
        for robot_id, robot in enumerate(robots):
            loss_avg += loss_sum[robot_id]
        loss_avg = loss_avg/len(robots)
        # loss_values.append((episode, loss_sum / tracker))
        loss_values.append((episode, loss_avg / tracker))

        # logging.info(f"The loss for episode: {episode} is {loss_sum / tracker}")
        logging.info(f"The loss for episode: {episode} is {loss_avg / tracker}")

    if episode % 10 == 0:
        for model_id, model in enumerate(models):
            model_targets[model_id].load_state_dict(model.state_dict())


pd.DataFrame(all_data, columns=["Episode", "Completed", "Epsilon", "Rewards", "Steps", "Loss"]).to_csv("resultsData.csv", index=False)
pd.DataFrame(episode_coordinates, columns=["Episode", "Coordinates"]).to_csv("EpisodeCoordinates.csv", index=False)

#
# pd.DataFrame(percentage_covered, columns=["Episode", "Percentage Completed"]).to_csv("PercentageCovered.csv", index=False)
# pd.DataFrame(loss_values, columns=["Episode", "Loss"]).to_csv("LossValues.csv", index=False)
# pd.DataFrame(episode_coordinates, columns=["Episode", "Coordinates"]).to_csv("EpisodeCoordinates.csv", index=False)
# pd.DataFrame(epsilon_tracker, columns=["Episode", "Epsilon"]).to_csv("EpsilonTracker.csv", index=False)
# pd.DataFrame(reward_tracker, columns=["Episode", "Reward"]).to_csv("RewardTracker.csv", index=False)
#
