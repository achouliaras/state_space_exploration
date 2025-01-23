import torch
import numpy as np
import random

import torch.backends
import torch.backends.cudnn

def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RNG_Agent:
    def __init__(self, name, action_space, cfg):
        """Initialize an agent that takes random actions
        """
        self.name = 'RANDOM' if name == None else name
        self.action_space = action_space
        print(action_space)
        self.cfg = cfg
        self.action_queue = []

    def get_action(self):
        action = self.action_space.sample()
        return action
                  
    def get_name(self):
        return self.name

