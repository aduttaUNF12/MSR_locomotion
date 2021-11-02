import os.path
import sys
import argparse

from tools.networks import CNN, FCNN
from tools.constants import MAX_EPISODE, NUM_MODULES, EPISODE, LEADER_ID, NN_TYPE
from tools.robot_interface import Module
from tools.loggers import path_maker


__version__ = '10.26.21'
# TODO: add more documentation


if __name__ == '__main__':
    if NN_TYPE == "FCNN":
        target_net = FCNN(NUM_MODULES, 0.001, 3)
        policy_net = FCNN(NUM_MODULES, 0.001, 3)
    elif NN_TYPE == "CNN":
        target_net = CNN(NUM_MODULES, 0.001, 3)
        policy_net = CNN(NUM_MODULES, 0.001, 3)
    else:
        print("Enter the type of network to be used!!", file=sys.stderr)
        exit(1)

    import time
    start_time = time.time()
    print(f"Starting the training ({NN_TYPE}) : {time.strftime('%H:%M:%S', time.localtime())}")
    eps_history = []
    filename = "null"

    module = Module(target_net, policy_net)
    assign_ = False
    learn = True
    i = 0
    episode_lens = []
    while i < 100:
        i += 1
        time.sleep(0.05)
    print(f"Finished buffer period in: {time.time()-start_time} ===== {time.strftime('%H:%M:%S', time.localtime())}")

    last_episode = time.time()
    total_elapsed_time = 0

    while module.step(module.timeStep) != -1:
        i = 0
        while i < 1000:
            i += 1

        module.learn()
        module.t += module.timeStep / 1000.0
        total_elapsed_time += module.timeStep / 1000.0
        module.total_time_elapsed = total_elapsed_time

        # if 0 <= total_elapsed_time % (60/5) <= 1:
        if 0 <= total_elapsed_time % 60 <= 1:
            if not assign_:
                EPISODE += 1
                module.episode = EPISODE

                if module.bot_id == LEADER_ID:
                    temp = time.time() - last_episode
                    print(f"Episode: {EPISODE} -- "
                          f"{time.time() - start_time} ===== time since last episode: {temp} ====== Episode reward: {module.episode_reward} == Loss: {module.loss}")
                    episode_lens.append(temp)

                assign_ = True
                module.simulationReset()
                module.old_pos = module.gps.getValues()
                last_episode = time.time()
        else:
            # assign_ is a temp needed to prevent infinite loop on the first Episode
            assign_ = False

        # if EPISODE > MAX_EPISODE:
        if EPISODE > MAX_EPISODE:
            end = time.time()
            if module.bot_id == LEADER_ID:
                print(f"Ending training of {NN_TYPE}.")
                print(f"Avg time per episode: {float(sum(episode_lens)/len(episode_lens))} sec")
                temp = end - start_time
                print(f"Runtime: {temp} secs")
                print(f"Runtime: {temp/60} mins")
                print(f"Runtime: {temp/60/60} hours")
                print("LOGGING OFF")
                with open(os.path.join(path_maker(), "time_stats.txt"), "w") as fin:
                    fin.write(f"Runtime: {temp} secs\nRuntime: {temp/60} mins\nRuntime: {temp/60/60} hours\n")
            module.simulationQuit(0)
            exit(0)
