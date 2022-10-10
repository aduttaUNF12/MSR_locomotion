N = 16
NUM_MODULES = 4
state_size = 5*N*N
action_count = 5
embedding_size = 8
# buffer = 100000
buffer = 10000
sample_length = 32

eps_start = 1.0
EPS = eps_start
eps_end = 0.01
eps_decay = 300
gamma = 0.99
learning_rate = 0.001

EPISODES = 2000
# EPISODES = 200
EXPLORE = EPISODES * 0.2
# EXPLORE = 20
