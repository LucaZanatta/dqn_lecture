import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from src import NetworkCartPole


def main():
    path = Path("data/2023-01-11_16-49-25/net_best.pth")
    # path = Path("data/2023-01-11_11-14-19/net_chkpt_00000.pth")

    games = 100

    env = gym.make("CartPole-v1", render_mode='human')
    device = torch.device("cpu")

    # Network
    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    net = NetworkCartPole(
        input_dim = input_dim,
        output_dim = output_dim,
        lr=1e-2,
        device=device
    )
    net.load(path)

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
            action = torch.argmax(net(torch.tensor(state))).item()
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            cntr_steps += 1
            sum_rewards += reward

        cntr_steps_list.append(cntr_steps)
        rewards_list.append(sum_rewards)
        rewards_mean.append(np.mean(rewards_list[-100:]))

        print(
            f"[game {game}]\tsteps: {cntr_steps:3.0f}\t" + \
                f"reward: {sum_rewards:4.2f}\t" + \
                    f"avg rewards: {np.mean(rewards_list[-100:]):4.2f}\t"
        )
    
    plt.figure()
    plt.plot(rewards_mean)
    plt.xlabel("# Games")
    plt.ylabel("Mean Score")
    plt.title("Mean scores over the games")
    plt.show()

if __name__ == "__main__":
    main()