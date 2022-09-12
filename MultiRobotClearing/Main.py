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

for r_id, r in enumerate(range(NUM_MODULES)):
    robots.append(Agent(r_id+1))

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
    obs = []
    act = []
    n_obs = []
    rwd = []
    macts = []
    n_macts = []
    # observations
    for p, m in enumerate(memory[0]):
        # making env[1] have all friendlies (so removing this robot's position)
        # m[r_id][0][1][memory[5][p][r_id][0]][memory[5][p][r_id][1]] = 0
        # making env[3] have me
        # m[r_id][0][3][memory[5][p][r_id][0]][memory[5][p][r_id][1]] = 1
        obs.append(m[r_id])
    # actions
    for m in memory[1]:
        act.append(m[r_id])
    # previous observations
    for p, m in enumerate(memory[2]):
        # making env[1] have all friendlies
        # m[r_id][0][1][memory[6][p][r_id][0]][memory[6][p][r_id][1]] = 0
        # making env[3] have me
        # m[r_id][0][3][memory[6][p][r_id][0]][memory[6][p][r_id][1]] = 1
        n_obs.append(m[r_id])
    # rewards
    for m in memory[3]:
        rwd.append(m[r_id])
    # mean actions
    for m in memory[4]:
        macts.append([m[r_id]])
    # previous mean actions
    for m in memory[7]:
        n_macts.append([m[r_id]])

    obs = torch.cat(obs).to(device)
    act = torch.cat(act).to(device)
    n_obs = torch.cat(n_obs).to(device)
    rwd = torch.tensor(rwd).to(device)
    macts = torch.tensor(macts).to(device)
    n_macts = torch.tensor(n_macts).to(device)
    obs.requires_grad = True
    n_obs.requires_grad = True

    q_values = model(obs, macts).gather(1, act).to(device)

    target_q_values = model_target(n_obs, n_macts).max(1)[0].detach().to(device)

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
    for robot in robots:
        if robot.agent_id == 1:
            main_environment.set_personal_pos(robot, x=0, y=0, reset=True)
        elif robot.agent_id == 2:
            main_environment.set_personal_pos(robot, x=0, y=N-1, reset=True)
        elif robot.agent_id == 3:
            main_environment.set_personal_pos(robot, N-1, y=0, reset=True)
        elif robot.agent_id == 4:
            main_environment.set_personal_pos(robot, x=N-1, y=N-1, reset=True)


restart()

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

    done = False
    hidden = None
    step_count = 0
    tracker = 0

    # Initial Environment reset and robot placement
    main_environment.reset()
    restart()

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
            m_a = torch.tensor([mean_actions[robot.agent_id]]).view(1,1)
            # getting state based on the current robot
            observation = main_environment.get_state(robot)
            # flattening the state
            observation = torch.tensor(observation).view(1, 4, N, N)
            # getting action and new EPS value
            action, EPS = robot.select_action(observation, model,
                                              main_environment, EPS, mean_action=torch.tensor(m_a).view(1,1))
            # taking a step for current robot
            next_observation, old_x, reward, done, old_y = main_environment.step(action, robot)
            observations_n.append(next_observation.view(1, 4, N, N))
            coordinates_n.append((robot.x_coordinate, robot.y_coordinate))
            actions.append(action)
            observations.append(observation.view(1, 4, N, N))
            rewards.append(reward)
            total_steps += 1
            reward = torch.tensor([reward])
            observation = next_observation
            # keep track of episodic reward
            ep_rwd[robot.agent_id] += reward
            coordinates[robot.agent_id].append((robot.x_coordinate, robot.y_coordinate))
            step_count += 1
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
        # this prevents the expbuffer bug
        for robot in robots:
            temp_mean_action.append(mean_actions[robot.agent_id])
            temp_mean_action_n.append(mean_actions_n[robot.agent_id])

        main_memory.push(observations, observations_n, actions, rewards, temp_mean_action, coordinates_, coordinates_n, temp_mean_action_n)
        del temp_mean_action, temp_mean_action_n
        mean_actions_n.clear()
        if t == 200:
            done = True
        if done:
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
            break

        if episode > EXPLORE:
            observations, next_observations, actions, rewards, mean_actions_, coordinates__, coordinates__n, mean_actions__n = main_memory.sample_with_batch(sample_length)
            # possible indexing of named tuples
            for robot_id, robot in enumerate(robots):
                push_data = [observations, actions, next_observations,
                             rewards, mean_actions_, coordinates__, coordinates__n, mean_actions__n]
                loss, eps = optimize_model(episode, model, model_target,
                                           adam, push_data, scheduler, robot_id)
                EPS = eps
                loss_sum[robot_id+1] += float(loss)
                tracker = t

    if episode > EXPLORE:
        loss_avg = 0
        for robot_id, robot in enumerate(robots):
            loss_avg += loss_sum[robot_id+1]
        loss_avg = loss_avg/len(robots)
        loss_values.append((episode, loss_avg / tracker))
        logging.info(f"The loss for episode: {episode} is {loss_avg / tracker}")

    if episode % 10 == 0:
        model_target.load_state_dict(model.state_dict())


pd.DataFrame(all_data, columns=["Episode", "Completed", "Epsilon", "Rewards", "Steps", "Loss"]).to_csv("resultsData.csv", index=False)
pd.DataFrame(episode_coordinates, columns=["Episode", "Coordinates"]).to_csv("EpisodeCoordinates.csv", index=False)
