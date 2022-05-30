import torch
import numpy as np
from collections import namedtuple
import random
import numpy as np
from collections import deque

class ExpBuffer():
    def __init__(self, max_storage, length):
        self.max_storage = max_storage
        self.length = length
        self.memory = deque([], maxlen=max_storage)
        self.Transitions = namedtuple('Transition', ('observation', 'next_observation', 'action', 'reward', 'mean_action'))
        self.steps = []
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]
    #   return torch.tensor(last_actions), torch.tensor(last_observations), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(observations), torch.tensor(dones)
    
    def push(self, *args):
        self.memory.append(self.Transitions(*args))
    
    def add_to_memory(self, episode):
        if episode > len(self.memory) - 1:
            self.memory.pop()
        self.memory.append(self.steps)
        self.steps = []
    
    def sample_with_batch(self, batch_size, seed=None):
        if seed:
            random.seed(seed)
        batch = random.sample(self.memory, batch_size)
        batch = self.Transitions(*zip(*batch))
        return batch
    """
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    """
    
    def get_sample(self, batch_size):
        return len(self.memory) >= batch_size