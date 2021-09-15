import numpy as np


# FROM MFRL paper
class ReplayBuffer:
    """a circular queue based on numpy array, supporting batch put and batch get"""
    def __init__(self, shape, dtype=np.float32):
        self.buffer = np.empty(shape=shape, dtype=dtype)
        self.head   = 0
        self.capacity   = len(self.buffer)
        self.return_buffer_len = 0

    def put(self, data):
        """put data to
        Parameters
        ----------
        data: numpy array
            data to add
        """
        head = self.head
        n = data.size
        if head + n <= self.capacity:
            self.buffer[head:head+n] = data
            self.head = (self.head + n) % self.capacity
        else:
            split = self.capacity - head
            self.buffer[head:] = data[:split]
            self.buffer[:n - split] = data[split:]
            self.head = split
        return n

    def get(self, index):
        """get items
        Parameters
        ----------
        index: int or numpy array
            it can be any numpy supported index
        """
        return self.buffer[index]

    def clear(self):
        """clear replay buffer"""
        self.head = 0
