from datetime import date


# Constants
EPSILON = 1
GAMMA = 0.1
MIN_EPSILON = 0.1
T = 0.1
# BATCH_SIZE = 32
BATCH_SIZE = 5
MAX_EPISODE = 20000
# MAX_EPISODE = 5000
EPISODE_LIMIT = 5010   # Limiting Replay Memory to 200 Episodes

# TODO: make capacity 200k
# MEMORY_CAPACITY = 10000  # this equals to a memory pool of over 4 mil spots, and with 5000 episodes each having 577
MEMORY_CAPACITY = 2**22  # this equals to a memory pool of over 4 mil spots, and with 5000 episodes each having 577
#                       of data we would only need just over 3 mil spots, so this should work and optimize things a bit
UPDATE_PERIOD = 10
EPISODE = 0
DATE_TODAY = date.today()

TOTAL_ELAPSED_TIME = 0
LEADER_ID = 1
TRIAL_PATH = ""
TRIAL_NAME = ""
REPLAY_MEMORY_EPISODE = 1


BASE_LOGS_FOLDER = None
# NN_TYPE = None
# POLICY_NET = None
# TARGET_NET = None

NUM_MODULES = 3
INITIAL = [0, 0.5, 0]

RE_ADJUST = False




