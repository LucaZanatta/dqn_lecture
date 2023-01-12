import numpy as np
from typing import Union


class EpsilonGreedy:
    """This class takles the exploration-exploitation dilemma with the epsilon greedy strategy"""

    def __init__(self, eps_start: int, eps_min: int, eps_decay: int, actions: int) -> None:
        """
        Creates the attributes

        Args:
            eps_start: starting probability of choosing a random action (usually is 1)
            eps_min: the minimun value of choosing a random action (greater than 0)
            eps_decay: the decay rate of the eps
            actions: number of actions
        """
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.actions = actions
            
    def _decay(self) -> None:
        """Decay the eps if it is greater the eps_min"""
        self.eps *= self.eps_decay
        if self.eps < self.eps_min:
            self.eps = self.eps_min

    def random_action(self) -> Union[int, None]:
        """
        Chooses if the there is a random action or not and it decays the eps.

        Returns:
            action: if None, no random action is performed otherwise action is an int.
        """
        action = np.random.choice(self.actions) if self.eps > np.random.uniform() else None
        self._decay()
        return action
