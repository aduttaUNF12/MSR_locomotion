import sys

from tools.networks import CNN, FCNN
from tools.constants import *
from tools.robot_interface import Module


__version__ = '09.13.21'
# TODO: fix the modular seperation and imports
# TODO: fix the buffers
# TODO: fix the constants.py
# TODO: add more documentation
if __name__ == '__main__':
    # global NN_TYPE
    # TODO: for now the NN selector is a string variable
    # NN_TYPE = "FCNN"
    NN_TYPE = "CNN"

    # TODO: make selection for regular buffer and priority buffer

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

    module = Module(target_net, policy_net, NN_TYPE)
    assign_ = False
    learn = True
    i = 0
    episode_lens = []
    while i < 100:
        i += 1
        time.sleep(0.05)
    print(f"Finished buffer period in: {time.time()-start_time} ===== {time.strftime('%H:%M:%S', time.localtime())}")

    last_episode = time.time()
    while module.step(module.timeStep) != -1:
        i = 0
        while i < 1000:
            i += 1

        module.learn()
        module.t += module.timeStep / 1000.0
        TOTAL_ELAPSED_TIME += module.timeStep / 1000.0
        module.total_time_elapsed = TOTAL_ELAPSED_TIME
        if 0 <= TOTAL_ELAPSED_TIME % 60 <= 1:
        # if 0 <= TOTAL_ELAPSED_TIME % 30 <= 1:
        # if 0 <= TOTAL_ELAPSED_TIME % 5 <= 1:
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
            exit()
