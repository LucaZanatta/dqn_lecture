import torch
from torch import optim
from torch import nn
from pathlib import Path
from typing import Tuple


class NetworkCartPole(nn.Module):
    """Network used in the CartPole task"""

    def __init__(self, input_dim: Tuple, output_dim: Tuple, lr: float, device: torch.device) -> None:
        """
        Creates the network, the optimized, the loss function and the attributes

        Args:
            input_dim: input dimension of the network
            output_dim: output dimension of the network
            lr: learning rate
            device: the device in which runs the code. IT WORKS ONLY IN CPU
        """
        super(NetworkCartPole, self).__init__()
        self.input_dim = input_dim[0]
        self.output_dim = output_dim
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, state: torch.tensor) -> torch.tensor:
        """
        The inference of the network.

        Returns:
            prediction of the network.
        """
        return self.net(state)

    def learn(self, y: torch.tensor, y_hat: torch.tensor) -> None:
        """
        In this method, the network compute the loss and the backpropagation

        Args:
            y: prediction
            y_hat: target 
        """
        loss = self.loss(y, y_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path: Path) -> None:
        """Saves the network"""
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        """Loads the network"""
        self.load_state_dict(torch.load(path))

