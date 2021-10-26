from datetime import date
import math

# NN params
# NN_TYPE = "FCNN"
NN_TYPE = "CNN"
EPSILON = 1
GAMMA = 0.1
MIN_EPSILON = 0.1
T = 0.1

# Episode params
EPISODE = 0  # starting Episode number
# MAX_EPISODE = 20000
MAX_EPISODE = 7000
# MAX_EPISODE = 5000
UPDATE_PERIOD = 10

# Batch params
BATCH_SIZE = 32
BATCH_PERCENT = 0.1
BUFFER_LIMIT = int(MAX_EPISODE*BATCH_PERCENT) \
    if math.modf(float(MAX_EPISODE*BATCH_PERCENT))[0] == 0\
    else int(math.modf(float(MAX_EPISODE*BATCH_PERCENT))[1])

# Time params
DATE_TODAY = date.today()

# Robot params
LEADER_ID = 1
NUM_MODULES = 3
