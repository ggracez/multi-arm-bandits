from environment import Environment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class eGreedy:

    def __init__(self, environment, epsilon=0) -> None:
        self.environment = environment
        self.epsilon = epsilon
        self.estimates = np.zeros(self.environment.arms)  # estimated q
        self.times_taken = np.zeros(self.environment.arms)  # n

    def __str__(self) -> str:
        if self.epsilon == 0:
            return f"ε = 0 (greedy)"
        else:
            return f"ε = {self.epsilon}"

    def choose_action(self) -> int:
        """either greedy (exploit) or epsilon (explore)

        Returns:
            int: the best action (which arm to pull)
        """
        if np.random.random() < self.epsilon:  # explore
            action = np.random.choice(len(self.estimates))  # = np.random.choice(self.environment.arms)

        else:  # exploit
            greedy_action = np.argmax(self.estimates)

            # find actions with same value as greedy action
            action = np.where(self.estimates == greedy_action)[0]  # returns list of actions

            # choose one of them at random (accounts for duplicates)
            if len(action) == 0:  # idk why but without this it breaks
                action = greedy_action
            else:
                action = np.random.choice(action)

        return action

    def update_estimates(self, reward: float, action) -> None:
        """update rule:
            q_(n+1) = q_n + 1/n * (r_n - q_n)
            estimate += (reward - estimate) / n

        Args:
            reward (float): nth reward
            action (int): nth action
        """
        self.times_taken[action] += 1  # update n
        self.estimates[action] += (reward - self.estimates[action]) / self.times_taken[action]  # update q

    def reset(self) -> None:
        """Reset to initial values
        """
        self.estimates = np.zeros(self.environment.arms)
        self.times_taken = np.zeros(self.environment.arms)


# actually run the bandit problems
def run_experiment(agents: list[eGreedy], environment, steps=1000) -> None:
    """each agent represents a different epsilon value

    Args:
        agents (list[eGreedy]): eGreedy agents with different epsilon values
        environment (Environment): defaults to 10-arm testbed with 2000 runs
        steps (int, optional): _description_. Defaults to 1000.
    """
    average_reward = np.zeros((steps, len(agents)))
    optimal_pulls = np.zeros((steps, len(agents)))

    for run in range(environment.runs):

        if run % 100 == 0:
            print(".", end="")

        environment.reset()
        for agent in agents:
            agent.reset()

        for step in range(steps):
            for i in range(len(agents)):  # for each epsilon value
                agent = agents[i]
                action = agent.choose_action()
                reward = np.random.normal(environment.means[action], 1)
                agent.update_estimates(reward, action)

                # average reward
                average_reward[step, i] += reward  # = arr[row][col]

                # % optimal action
                if action == environment.opt:
                    optimal_pulls[step, i] += 1

    # average the values
    average_reward /= environment.runs
    optimal_pulls /= environment.runs

    graph_results(average_reward, optimal_pulls, agents)


def graph_results(average_reward, optimal_pulls, eps: list[eGreedy]) -> None:
    """Graph results based on Figure 2.2 of the textbook

    Args:
        average_reward (numpy.ndarray): xy plot for number of steps vs average reward
        optimal_pulls (numpy.ndarray): xy plot for number of steps vs % optimal action
        eps (list[eGreedy]): different epsilon values
    """

    fig, (ax1, ax2) = plt.subplots(2)

    # Graph 2.2: Average Reward
    ax1.plot(average_reward)
    ax1.set_ylabel("Average Reward")
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Steps")
    ax1.legend(eps, loc="lower right")

    # Graph 2.2: % Optimal Action
    ax2.plot(optimal_pulls)

    # add percent symbols to y axis    
    yticks = mtick.PercentFormatter(xmax=1)
    ax2.yaxis.set_major_formatter(yticks)
    ax2.set_ylim(0, 1)

    ax2.set_ylabel("% Optimal Action")
    ax2.set_xlabel("Steps")
    ax2.legend(eps, loc="lower right")

    plt.show()
    # fig.savefig("figures/2.2_comparison.png")


def main():
    environment = Environment()  # can change # of runs here (default 2000)
    agents = []
    epsilon_vals = [0.1, 0.01, 0]
    for val in epsilon_vals:
        agents.append(eGreedy(environment, val))
    # run the experiment!!
    print("Running Experiment...")
    run_experiment(agents, environment)  # can change # of steps here (default 1000)
    print()


if __name__ == "__main__":
    main()
