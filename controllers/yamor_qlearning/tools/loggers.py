import math
import os
from .constants import DATE_TODAY, NUM_MODULES,EPSILON, MIN_EPSILON


def path_maker():
    log_path = os.path.join(os.getcwd(), "LOGS")
    if not os.path.isdir(log_path):
        try:
            os.mkdir(log_path)
        except FileExistsError:
            pass

    set_folder_path = os.path.join(log_path, "{}_MODULES".format(NUM_MODULES))
    if not os.path.isdir(set_folder_path):
        try:
            os.mkdir(set_folder_path)
        except FileExistsError:
            pass

    current_run = os.path.join(set_folder_path, "{}_RUN".format(DATE_TODAY))
    if not os.path.isdir(current_run):
        try:
            os.mkdir(current_run)
        except FileExistsError:
            pass

    return current_run


# logs the information throughout the trial run, collects time_step, reward, loss, and Episode number
def writer(name, num_of_bots, time_step, reward, loss, episode, nn_type, epsilon):
    current_run = path_maker()

    file_name = "{}_{}_MODULES_{}.txt".format(nn_type, num_of_bots, name)
    file_path = os.path.join(current_run, file_name)
    with open(file_path, "a") as fin:
        fin.write('{},{},{},{},{}\n'.format(time_step, reward, loss, episode, epsilon))


def logger(**kwargs):
    with open("log.txt", "a") as fin:
        for kwarg in kwargs:
            fin.write("{}: {}      ".format(kwarg, kwargs[kwarg]))
        fin.write("\n")
        fin.write("=================== ENTRY END ========================\n")

    kl = len(kwargs)
    with open("log.csv", "a") as fin:
        for p, kwarg in enumerate(kwargs):
            if (p+1) >= kl:
                fin.write("{}".format(kwargs[kwarg]))
            else:
                fin.write("{},".format(kwargs[kwarg]))
        fin.write("\n")
