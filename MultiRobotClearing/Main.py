from Config import (N, NUM_MODULES, action_count, buffer,
                    sample_length, eps_start, eps_end, eps_decay,
                    learning_rate, gamma, EPISODES, EXPLORE, EPS)


from Environment import Environment
from ExpBuffer import ExpBuffer
from DRQN import Model
from Agent import Agent
from Plotter import Plotter
from itertools import count
import torch
import math
import pandas as pd
import time
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename="Project2.log", filemode="w", level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# houses robots involved
robots = []
# Main environment
main_environment = Environment(N, action_count)
memories = []
for r_id, r in enumerate(range(NUM_MODULES)):
    robots.append(Agent(r_id+1))
    memories.append(ExpBuffer(buffer, sample_length))

main_memory = ExpBuffer(buffer, sample_length)


model = Model().double().to(device)
model_target = Model().double().to(device)
model_target.load_state_dict(model.state_dict())
model_target.eval()
adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, 'min')


# TODO: for sure rework this
def optimize_model(episode, model, model_target, adam, memory, scheduler, r_id):
    EPS = eps_end + (eps_start - eps_end) * \
          math.exp(-1. * (episode - EXPLORE) / eps_decay)
    difference_function = torch.nn.MSELoss()
    # memory = [observations, actions, next_observations, rewards, mean_actions]


    obs = torch.cat(memory[0]).to(device)
    n_obs = torch.cat(memory[2]).to(device)
    act = torch.cat(memory[1]).to(device)
    n_act = torch.tensor(memory[8]).to(device)
    rwd = torch.tensor(memory[3]).to(device)
    macts = torch.tensor(memory[4]).to(device)
    n_macts = torch.tensor(memory[7]).to(device)
    obs.requires_grad = True
    n_obs.requires_grad = True

    q_values = model(obs, act).gather(1, act).to(device)

    target_q_values = model_target(n_obs, n_act).max(1)[0].detach().to(device)

    final_results = rwd.double() + (gamma * target_q_values.double())

    adam.zero_grad()


    difference = difference_function(q_values, final_results.unsqueeze(-1).detach())
    difference.backward()

    adam.step()
    scheduler.step(difference)

    return difference, EPS



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
main_environment.reset()


# resets robot positions
def restart():
    main_environment.reset()
    for robot in robots:
        if robot.agent_id == 1:
            main_environment.reset(robot, x=0, y=0)
        elif robot.agent_id == 2:
            main_environment.reset(robot, x=0, y=N-1)
        elif robot.agent_id == 3:
            main_environment.reset(robot, N-1, y=0)
        elif robot.agent_id == 4:
            main_environment.reset(robot, x=N-1, y=N-1)


restart()
overall_steps = 0

robot_coordinates = {
    1: [],
    2: [],
    3: [],
    4: [],
}
robot_coordinates_episode = {

}

for episode in range(EPISODES):
    torch.cuda.empty_cache()    # for optimization of the code

    # reward tracker for all robots
    ep_rwd = {}
    for robot in robots:
        ep_rwd[robot.agent_id] = 0.0
    # loss tracker for all robots
    loss_sum = {}
    for robot in robots:
        loss_sum[robot.agent_id] = 0.0
    # coordinate tracker for all robots
    coordinates = {}
    for robot in robots:
        coordinates[robot.agent_id] = []
    # mean action tracker for all robots
    mean_actions = {}
    for robot in robots:
        mean_actions[robot.agent_id] = 0.0
    # previous mean action tracker
    mean_actions_n = {}
    for robot in robots:
        mean_actions_n[robot.agent_id] = 0.0
    # action tracker for all robots
    actions_n = {}
    for robot in robots:
        actions_n[robot.agent_id] = 0.0

    done = False
    hidden = None
    step_count = 0
    tracker = 0

    # Initial Environment reset and robot placement
    restart()
    # print(main_environment.env[0])
    # print(main_environment.env[1])
    # print(main_environment.env[2])
    # print(main_environment.env[3])
    # print(main_environment.env[4])
    # exit(1)
    # print("****************************************")
    # print("RESET COORDINATES")
    # for robot in robots:
    #     print(f"{robot.agent_id} -- ({robot.x_coordinate}, {robot.y_coordinate})")
    # print("****************************************")
    plot = None
    # plot = Plotter(N, environment.environment, environment.environment[0])

    for t in count():
        actions = []
        observations_n = []
        observations = []
        rewards = []
        robot_positions = []
        coordinates_ = []
        coordinates_n = []

        for robot in robots:
            # getting mean action of the robot
            if NUM_MODULES >= 2:
                m_a = torch.tensor([mean_actions[robot.agent_id]]).view(1,1)
            else:
                m_a = torch.tensor([0.0]).view(1,1)
            # getting state based on the current robot
            observation = main_environment.get_state(robot)
            # flattening the state
            observation = torch.tensor(observation).view(1, 4, N, N)
            # getting action and new EPS value

            robot_coordinates[robot.agent_id].append((robot.y_coordinate, robot.x_coordinate))
            action, EPS = robot.select_action(observation, model,
                                              main_environment, EPS, mean_action=torch.tensor(m_a).view(1,1))
            # taking a step for current robot
            # print("Env 0")
            # print(main_environment.env[0])
            # print("Env 1")
            # print(main_environment.env[1])
            # print("Env 2")
            # Maybe it should keep track of all previous robot locations instead of all
            # print(main_environment.env[2])
            # print("Env 3")
            # print(main_environment.env[3])
            # print("Env 4")
            # print(main_environment.env[4])
            # print(f"step for {robot.agent_id}")
            next_observation, old_x, reward, done, old_y = main_environment.step(action, robot)


            observations_n.append(next_observation.view(1, 4, N, N))

            coordinates_n.append((robot.y_coordinate, robot.x_coordinate))
            actions.append(action)

            observations.append(observation.view(1, 4, N, N))

            rewards.append(reward)
            total_steps += 1
            reward = torch.tensor([reward])
            observation = next_observation
            # keep track of episodic reward
            ep_rwd[robot.agent_id] += reward

            coordinates[robot.agent_id].append((robot.y_coordinate, robot.x_coordinate))

        step_count += 1
        overall_steps += 1
        # saving previous mean actions
        if len(robots) >= 2:
            for i, d in enumerate(robots):
                mean_actions_n[i+1] = mean_actions[i+1]

        # calculating mean action
        mean_actions.clear()
        if len(robots) >= 2:
            for i, d in enumerate(actions):
                temp = 0.0
                for ii, di in enumerate(actions):
                    if i != ii:
                        temp += actions[ii]
                temp /= len(actions)-1
                # mean_actions[i+1] = torch.tensor([temp]).view(1,1)
                mean_actions[i+1] = temp

        # plot.graph(episode, t)
        temp_mean_action = []
        temp_mean_action_n = []
        temp_action_n = []
        # this prevents the expbuffer bug
        for robot in robots:
            if NUM_MODULES >= 2:
                temp_mean_action.append(mean_actions[robot.agent_id])
                temp_mean_action_n.append(mean_actions_n[robot.agent_id])
            else:
                temp_mean_action.append(0.0)
                temp_mean_action_n.append(0.0)
            temp_action_n.append([actions_n[robot.agent_id]])

        for robot in robots:
            memories[robot.agent_id-1].push(observations[robot.agent_id-1], observations_n[robot.agent_id-1],
                                            actions[robot.agent_id-1], rewards[robot.agent_id-1],
                                            temp_mean_action[robot.agent_id-1], coordinates_, coordinates_n,
                                            temp_mean_action_n[robot.agent_id-1], temp_action_n[robot.agent_id-1])

        main_memory.push(observations, observations_n, actions, rewards, temp_mean_action, coordinates_, coordinates_n, temp_mean_action_n, temp_action_n)

        for p, robot in enumerate(robots):
            actions_n[robot.agent_id] = actions[p]

        del temp_mean_action, temp_mean_action_n, temp_action_n
        mean_actions_n.clear()

        if t == 300:
            done = True
        if done:
            robot_coordinates_episode[episode] = robot_coordinates
            robot_coordinates = {
                1: [],
                2: [],
                3: [],
                4: [],
            }
            percentage_covered.append((episode, main_environment.p_completion))
            episode_coordinates.append((episode, coordinates))
            epsilon_tracker.append((episode, EPS))
            # reward average
            ep_rwd_avg = 0.0
            for robot in robots:
                ep_rwd_avg += ep_rwd[robot.agent_id]
            ep_rwd_avg = ep_rwd_avg/len(robots)

            reward_tracker.append((episode, float(ep_rwd_avg)))
            # loss average
            ep_loss_avg = 0.0
            for robot in robots:
                ep_loss_avg += loss_sum[robot.agent_id]
            ep_loss_avg = ep_loss_avg/len(robots)

            all_data.append((episode, main_environment.p_completion, EPS, int(ep_rwd_avg), t, ep_loss_avg))

            end = time.time()
            print(f"Comp. {main_environment.p_completion*100: .2f}%, ep.: {episode} st: {t}, EPS: {EPS: .2f}, T: {(end-start)/60: .2f} mins., R: {float(ep_rwd_avg)}")
            print(f"Total number of steps were {t}")


        # if episode > EXPLORE and done:
        if overall_steps > 100000 and done:
            batch = main_memory.sample_pos(batch_size=sample_length)

            # possible indexing of named tuples
            for robot_id, robot in enumerate(robots):
                observations, next_observations, actions, rewards, mean_actions_, coordinates__, coordinates__n, mean_actions__n, actions__n  = memories[robot_id].get_by_pos(batch)
                push_data = [observations, actions, next_observations,
                             rewards, mean_actions_, coordinates__, coordinates__n, mean_actions__n, actions__n]
                loss, eps = optimize_model(episode, model, model_target,
                                           adam, push_data, scheduler, robot_id)
                EPS = eps
                loss_sum[robot_id+1] += float(loss)
                tracker = t


        # if episode > EXPLORE and done:
        if overall_steps > 100000 and done:
            loss_avg = 0
            for robot_id, robot in enumerate(robots):
                loss_avg += loss_sum[robot_id+1]
            loss_avg = loss_avg/len(robots)
            loss_values.append((episode, loss_avg / tracker))
            logging.info(f"The loss for episode: {episode} is {loss_avg / tracker}")
            print(f"***The loss for episode: {episode} is {loss_avg / tracker}")

        if episode % 10 == 0:
            model_target.load_state_dict(model.state_dict())

        if t >= 300 or done:
            break

import json
with open("robot_actions.json", "w") as fout:
    json.dump(robot_coordinates_episode, fout)
pd.DataFrame(all_data, columns=["Episode", "Completed", "Epsilon", "Rewards", "Steps", "Loss"]).to_csv("resultsData.csv", index=False)
pd.DataFrame(episode_coordinates, columns=["Episode", "Coordinates"]).to_csv("EpisodeCoordinates.csv", index=False)
