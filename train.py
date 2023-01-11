import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path


from src import ReplyMemory, EpsilonGreedy, NetworkCartPole, LearnDQN, AgentOffPolicy

class Checkpoint:
    """Takes care of the checkpoints and the network saving."""

    def __init__(self, path: Path, n_games: int) -> None:
        """
        Sets the attributes and it makes the dir tree.

        Args:
            path: path to the saving dir
            n_games: after how many games the chkeckpoint is done.
        """

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.path = path / time_str
        self.path.mkdir(parents=True, exist_ok=True)
        self.n_games = n_games
        self.cntr = 0

        self.idx = 0
        self.reward = None
    
    def __call__(self, net: torch.nn.Module, reward: float) -> None:
        """
        Saves the best network and the network if the checkpoint is reached.

        Args:
            net: network to save.
            reward: the mean reward.
        """
        if self.reward is None:
            self.reward = reward
        elif reward > self.reward:
            self.reward = reward
            self.save(net, "net_best.pth")
        elif self.cntr % self.n_games == 0:
            self.save(net, f"net_chkpt_{self.idx:05}.pth")
            self.idx += 1
        self.cntr +=1
    
    def save(self, net: torch.nn.Module, name: str) -> None:
        """
        Saves the network.
        
        Args:
            net: network to save
            name: name of the network.
        """
        net.save(self.path / name)


def main():
    games = 1000
    batch_sz = 32
    max_sz = 200
    batch_numbers = 1
    iterations_number = 1

    env = gym.make("CartPole-v1")  # TODO
    device = torch.device("cpu")

    # Checkpoint
    chkpt = Checkpoint(Path("data"), 20)
    
    # Network
    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    net = NetworkCartPole(
        input_dim = input_dim,
        output_dim = output_dim,
        lr=1e-2,
        device=device
    )

    # Memory
    memory = ReplyMemory(
        batch_sz=batch_sz,
        batch_numbers=batch_numbers,
        max_sz=max_sz,
        state_shape=input_dim,
        device=device,
    )

    # Exploration
    exploration = EpsilonGreedy(
        eps_start=1.,
        eps_min=.05,
        eps_decay=.9997,
        actions=output_dim,
    )

    # Algo
    learing_algo = LearnDQN(
        discount_factor=.99,
    )

    # Agent
    agent = AgentOffPolicy(
        net=net,
        memory=memory,
        exploration=exploration,
        learn_algo=learing_algo,
        iterations_number=iterations_number,
    )

    # some stats
    cntr_steps_list = []
    rewards_list = []
    rewards_mean = []

    for game in range(games):
        done = False
        sum_rewards = 0
        cntr_steps = 0

        state, _ = env.reset()
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            # done = done or truncated

            agent.memory_add(state, action, reward, state_, done)

            state = state_

            agent.learn()

            cntr_steps += 1
            sum_rewards += reward

        cntr_steps_list.append(cntr_steps)
        rewards_list.append(sum_rewards)
        rewards_mean.append(np.mean(rewards_list[-100:]))

        chkpt(agent.net, rewards_mean[-1])

        print(
            f"[game {game}]\tsteps: {cntr_steps:3.0f}\t" + \
                f"reward: {sum_rewards:4.2f}\t" + \
                    f"avg rewards: {np.mean(rewards_list[-100:]):4.2f}\t" + \
                        f"eps: {exploration.eps:.2f}"
        )
    
    np.save(chkpt.path / "rewards.npy" , np.array(rewards_list))
    np.save(chkpt.path / "rewards_mean.npy" , np.array(rewards_mean))

    plt.figure()
    plt.plot(rewards_mean)
    plt.xlabel("# Games")
    plt.ylabel("Mean Score")
    plt.title("Mean scores over the games")
    plt.savefig(chkpt.path / "plot.png")

if __name__ == "__main__":
    main()
