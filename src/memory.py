import numpy as np
import torch
from typing import Tuple, List


class ReplyMemory:
    def __init__(self, batch_sz: int, batch_numbers: int, max_sz: int, state_shape: tuple, device:torch.device) -> None:
        """
        Creates the attributes of the class.

        Args:
            batch_sz: batch size
            batch_numbers: number of batch in each "mini epoch"
            max_sz: size of the memory
            state_shape: shape of the state
            device: the device used. ONLY CPU WORKS.
        """

        self.batch_sz = batch_sz
        self.batch_numbers = batch_numbers
        self.max_sz = max_sz
        self.state_shape = state_shape
        self.device = device

        self.actions = torch.zeros(self.max_sz, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(self.max_sz, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.max_sz, dtype=torch.int8, device=self.device)
        self.states = torch.zeros((self.max_sz, *state_shape), dtype=torch.float32, device=self.device)
        self.states_ = torch.zeros((self.max_sz, *state_shape), dtype=torch.float32, device=self.device)
        
        self.idx_memory = 0

    def add(self, state: np.array, action: int, reward: float, state_: np.array, dones: int) -> None:
        """
        Adds the sample into the memory.

        Args:
            state: the state of the environment
            action: the action taken
            reward: the reward collected
            state_: the future state
            dones: if the game ends
        """
        
        idx = self.idx_memory % self.max_sz
        self.actions[idx] = action
        self.states[idx] = torch.tensor(state, dtype=torch.float32)
        self.rewards[idx] = reward
        self.states_[idx] = torch.tensor(state_, dtype=torch.float32)
        self.dones[idx] = dones

        self.idx_memory += 1

    def batch(self) -> Tuple[List[torch.tensor], List[torch.tensor], List[torch.tensor], List[torch.tensor], List[torch.tensor]]:
        """
        Creates the batches.

        Returns:
            states: batches of states
            actions: batches of actions
            rewards: batches of rewards
            states_: batches of the future states
            dones: batches of the ended episodes
        """

        batch_idxs = np.random.randint(self.max_sz, size=(self.batch_numbers, self.batch_sz))
        
        actions = []
        states = []
        rewards = []
        states_ = []
        dones = []

        for i in range(self.batch_numbers):
            actions.append(self.actions[batch_idxs[i]])
            states.append(self.states[batch_idxs[i]])
            rewards.append(self.rewards[batch_idxs[i]])
            states_.append(self.states_[batch_idxs[i]])
            dones.append(self.dones[batch_idxs[i]])
        
        return states, actions, rewards, states_, dones

    
