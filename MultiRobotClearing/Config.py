N = 16
# NUM_MODULES = 4
NUM_MODULES = 3
state_size = 5*N*N
action_count = 5
embedding_size = 8
buffer = 100000
sample_length = 64

eps_start = 0.9
EPS = eps_start
eps_end = 0.05
eps_decay = 250
gamma = 0.99
learning_rate = 0.001

EPISODES = 2000
# EPISODES = 200
EXPLORE = EPISODES * 0.2
