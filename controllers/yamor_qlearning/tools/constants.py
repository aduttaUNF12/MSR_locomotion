from datetime import date
import math

# NN params
NN_TYPE = "FCNN"
# NN_TYPE = "CNN"
EPSILON = 0.9
GAMMA = 0.1
MIN_EPSILON = 0.05
# MIN_EPSILON = 0.1
EPS_EXP = True
DECAY_PAUSE_EPISODE = 300

if EPS_EXP:
    # EPSILON_DECAY = 7000  # for 20k (35%)
    # EPSILON_DECAY = 1750  # for 5k (35%)
    # EPSILON_DECAY = 200  #
    EPSILON_DECAY = 600  #
else:
    # EPSILON_DECAY = 0.00226
    # EPSILON_DECAY = 0.005
    # EPSILON_DECAY = 0.006
    # EPSILON_DECAY = 0.0006
    EPSILON_DECAY = 0.00113
    # EPSILON_DECAY = 0.00000113
    # EPSILON_DECAY = 0.000000113
    # EPSILON_DECAY = 0.0000006
    # EPSILON_DECAY = 0.0000003
# T = 0.1
T = 0.59
COMMUNICATION = True

FIX = False


# Episode params
EPISODE = 0  # starting Episode number
# MAX_EPISODE = 20000
# MAX_EPISODE = 7000
# MAX_EPISODE = 5000
# MAX_EPISODE = 600
MAX_EPISODE = 2000
UPDATE_PERIOD = 10

# Batch params
REGULAR_BUFFER = False
BATCH_SIZE = 32
# BATCH_SIZE = 10
# BATCH_SIZE = 5
# BATCH_SIZE = 10
BATCH_PERCENT = 0.1
if not REGULAR_BUFFER:
    BUFFER_LIMIT = int(MAX_EPISODE*BATCH_PERCENT) \
        if math.modf(float(MAX_EPISODE*BATCH_PERCENT))[0] == 0\
        else int(math.modf(float(MAX_EPISODE*BATCH_PERCENT))[1])
else:
    BUFFER_LIMIT = MAX_EPISODE

# Time params
DATE_TODAY = date.today()

# Robot params
LEADER_ID = 1
# NUM_MODULES = 6
NUM_MODULES = 3
