from environment import Environment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class eGreedy:

    def __init__(self, environment, epsilon=0.0, stepsize=None):
        self.environment = environment
        self.epsilon = epsilon
        self.stepsize = stepsize
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

    def update_estimates(self, reward: float, action):
        """Update rule:
            estimate +=  stepsize * (reward - estimate)

        Args:
            reward (float): nth reward
            action (int): nth action
        """
        if self.stepsize:
            # constant stepsize: exponential recency weighted average (ERWA)
            self.estimates[action] += self.stepsize * (reward - self.estimates[action])
        else:
            # sample average stepsize: stepsize = 1/n
            self.times_taken[action] += 1  # update n
            self.estimates[action] += (reward - self.estimates[action]) / self.times_taken[action]  # update q

    def reset(self):
        """Reset to initial values
        """
        self.estimates = np.zeros(self.environment.arms)
        self.times_taken = np.zeros(self.environment.arms)


# actually run the bandit problems
def run_experiment(agents: list[eGreedy], environment, steps=1000):
    """each agent represents a different epsilon value

    Args:
        agents (list[eGreedy]): eGreedy agents with different epsilon values.
        environment (Environment): Defaults to 10-arm testbed with 2000 runs.
        steps (int, optional): Number of time steps per run. Defaults to 1000.

    Returns:
        average_reward (numpy.ndarray): xy plot for number of steps vs average reward
        optimal_pulls (numpy.ndarray): xy plot for number of steps vs % optimal action
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
            environment.update_arms()
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

    return average_reward, optimal_pulls


def graph_results(average_reward, optimal_pulls, legend, save_loc):
    """Graph results based on Figure 2.2 of the textbook

    Args:
        average_reward (list[numpy.ndarray]): list of xy plots for number of steps vs average reward
        optimal_pulls (list[numpy.ndarray]): list of xy plots for number of steps vs % optimal action
        legend (list[eGreedy | str]): legend for the graph
        save_loc (str): location to save the figure
    """

    fig, (ax1, ax2) = plt.subplots(2)

    # Graph 2.2: Average Reward
    for line in average_reward:
        ax1.plot(line)
    ax1.set_ylabel("Average Reward")
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Steps")
    ax1.legend(legend)

    # Graph 2.2: % Optimal Action
    for line in optimal_pulls:
        ax2.plot(line)

    # add percent symbols to y axis    
    y_ticks = mtick.PercentFormatter(xmax=1)
    ax2.yaxis.set_major_formatter(y_ticks)
    ax2.set_ylim(0, 1)

    ax2.set_ylabel("% Optimal Action")
    ax2.set_xlabel("Steps")
    ax2.legend(legend)

    plt.show()
    fig.savefig(save_loc)


def compare_stationary_eps():
    """Compare different epsilon values in a stationary environment
    """
    environment = Environment()  # can change # of runs here (default 2000)
    agents = []
    epsilon_vals = [0.1, 0.01, 0]
    for val in epsilon_vals:
        agents.append(eGreedy(environment, val))
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)  # can change # of steps here (default 1000)
    graph_results([average_reward], [optimal_pulls], agents, "figures/2.2_comparison.png")
    print()


def compare_envs():
    """Compare stationary and nonstationary environments with 10000 steps and 0.1 epsilon
    """
    stationary_env = Environment()
    nonstationary_env = Environment(arms=4, stationary=False, decay=0.05)
    print("Running Experiment...")
    s_reward, s_optimal = run_experiment([eGreedy(stationary_env, 0.1)], stationary_env, steps=10000)
    n_reward, n_optimal = run_experiment([eGreedy(nonstationary_env, 0.1)], nonstationary_env, steps=10000)
    print()

    average_reward = [s_reward, n_reward]
    optimal_pulls = [s_optimal, n_optimal]
    legend = ["Stationary", "Nonstationary"]
    save_loc = "figures/egreedy_environment_comparison.png"

    graph_results(average_reward, optimal_pulls, legend, save_loc)


def compare_ns_stepsizes():
    """Compare ERWA and sample average stepsize for non-stationary environment with 10000 steps and 0.1 epsilon
    """
    env = Environment(arms=4, stationary=False, decay=0.05)
    print("Running Experiment...")
    sa_reward, sa_optimal = run_experiment([eGreedy(env, 0.1, None)], env, steps=10000)
    env.reset()
    erwa_reward, erwa_optimal = run_experiment([eGreedy(env, 0.1, 0.1)], env, steps=10000)
    print()

    average_reward = [sa_reward, erwa_reward]
    optimal_pulls = [sa_optimal, erwa_optimal]
    legend = ["stepsize = 1/n", "stepsize = 0.1"]
    save_loc = "figures/nonstationary_stepsize_comparison.png"

    graph_results(average_reward, optimal_pulls, legend, save_loc)


def main():
    compare_stationary_eps()
    # compare_envs()
    # compare_ns_stepsizes()


if __name__ == "__main__":
    main()
