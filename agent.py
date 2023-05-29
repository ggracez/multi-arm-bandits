from environment import Environment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class Agent:

    def __init__(self, environment, epsilon=0.0, stepsize=None, ucb_param=None, gradient=False, baseline=False):

        # policy=[egreedy, ucb, gradient]
        # param = [epsilon, ucb_param, stepsize]
        self.environment = environment
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.c = ucb_param
        self.gradient = gradient
        self.baseline = baseline
        self.average_reward = 0

        self.estimates = np.zeros(self.environment.arms)  # estimated q
        self.times_taken = np.zeros(self.environment.arms)  # n
        self.action_prob = np.zeros(self.environment.arms)  # probability π
        self.time = 0  # t

    def __str__(self) -> str:
        if self.c:
            return f"UCB c = {self.c}"
        elif self.gradient:
            if self.baseline:
                return f"Gradient α = {self.stepsize} with baseline"
            else:
                return f"Gradient α = {self.stepsize}"
        elif self.epsilon == 0:
            return f"ε = 0 (greedy)"
        else:
            return f"ε = {self.epsilon}"

    def choose_action(self) -> int:
        """either greedy (exploit) or epsilon (explore)

        Returns:
            int: the best action (which arm to pull)
        """
        if self.c:
            # UCB
            # get UCB estimate (add 1e-5 to avoid divide by 0)
            ucb_estimate = self.estimates + (self.c * np.sqrt(np.log(self.time + 1) / (self.times_taken + 1e-5)))
            ucb_action = np.argmax(ucb_estimate)
            action = np.where(ucb_estimate == ucb_action)[0]
            if len(action) == 0:
                action = ucb_action
            else:
                action = np.random.choice(action)

        elif self.gradient:
            # soft-max distribution
            exponential = np.exp(self.estimates)
            self.action_prob = exponential / np.sum(exponential)
            action = np.random.choice(len(self.estimates), p=self.action_prob)

        elif np.random.random() < self.epsilon:  # explore
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
        self.times_taken[action] += 1  # update n
        self.time += 1  # update t
        self.average_reward += (reward - self.average_reward) / self.time

        if not self.stepsize:
            # sample average stepsize: stepsize = 1/n
            self.estimates[action] += (reward - self.estimates[action]) / self.times_taken[action]  # update q
        elif self.gradient:
            # gradient algorithm uses action preferences, updated with:
            #   pref += stepsize * (reward - baseline) * (1 - prob of action) for all selected actions and
            #   pref += stepsize * (reward - baseline) * (0 - prob of action) for all non-selected actions
            # baseline is average reward up to but not including current time step - very good if mean != 0
            #   if reward > baseline, then the probability of taking A_t in the future is increased and vice versa
            #   non-selected actions move in the opposite direction
            if self.baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            one_hot = np.zeros(self.environment.arms)
            one_hot[action] = 1
            self.estimates += self.stepsize * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # constant stepsize: exponential recency weighted average (ERWA)
            self.estimates[action] += self.stepsize * (reward - self.estimates[action])

    def reset(self):
        """Reset to initial values
        """
        self.estimates = np.zeros(self.environment.arms)
        self.times_taken = np.zeros(self.environment.arms)
        self.action_prob = np.zeros(self.environment.arms)
        self.time = 0


# actually run the bandit problems
def run_experiment(agents: list[Agent], environment, steps=1000):
    """each agent represents a different epsilon value

    Args:
        agents (list[Agent]): eGreedy agents with different epsilon values.
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


def graph_results(average_reward, optimal_pulls, title, legend, save_loc):
    """Graph results based on Figure 2.2 of the textbook

    Args:
        average_reward (list[numpy.ndarray]): list of xy plots for number of steps vs average reward
        optimal_pulls (list[numpy.ndarray]): list of xy plots for number of steps vs % optimal action
        title (str): title of the graph
        legend (list[Agent | str]): legend for the graph
        save_loc (str): location to save the figure
    """

    if average_reward and optimal_pulls:
        fig, (ax1, ax2) = plt.subplots(2)
    elif average_reward:
        fig, ax1 = plt.subplots(1)
    elif optimal_pulls:
        fig, ax2 = plt.subplots(1)

    fig.suptitle(title, fontsize=16)

    if average_reward:
        # Graph 2.2: Average Reward
        for line in average_reward:
            ax1.plot(line)
        ax1.set_ylabel("Average Reward")
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Steps")
        ax1.legend(legend)

    if optimal_pulls:
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
        agents.append(Agent(environment, epsilon=val))
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)  # can change # of steps here (default 1000)
    title = "Different Epsilon Values in a Stationary Environment"
    graph_results([average_reward], [optimal_pulls], title, agents, "figures/2.2_comparison.png")
    print()


def compare_stationary_ucb():
    environment = Environment()
    egreedy_agent = Agent(environment, epsilon=0.1)
    ucb_agent = Agent(environment, ucb_param=2)
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment([egreedy_agent, ucb_agent], environment)
    title = "eGreedy with ε=0.1 and UCB with c=2 in a Stationary Environment"
    legend = ["eGreedy", "UCB"]
    save_loc = "figures/2.4_egreedy_ucb_comparison.png"
    graph_results([average_reward], [optimal_pulls], title, legend, save_loc)
    print()


def compare_stationary_gradients():
    environment = Environment(mean=4)  # shift the mean to see effects of the baseline
    agents = []
    alpha_vals = [0.1, 0.4]
    for val in alpha_vals:
        agents.append(Agent(environment, stepsize=val, gradient=True))
        agents.append(Agent(environment, stepsize=val, gradient=True, baseline=True))
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)
    title = "Performance of Gradient Bandit Algorithms in a Stationary Environment"
    graph_results([], [optimal_pulls], title, agents, "figures/2.5_gradient_comparison.png")
    print()


def compare_stationary_all():
    environment = Environment()
    agents = [
        Agent(environment),  # pure greedy
        Agent(environment, epsilon=0.1),  # egreedy with epsilon=0.1
        Agent(environment, ucb_param=2),  # ucb with c=2
        Agent(environment, stepsize=0.1, gradient=True)  # gradient with alpha=0.1
    ]
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)
    title = "Performance of Various Bandit Algorithms in a Stationary Environment"
    graph_results([], [optimal_pulls], title, agents, "figures/bandits_comparison_stationary.png")
    print()


def compare_nonstationary_all():
    environment = Environment(arms=4, stationary=False, decay=0.05)
    agents = [
        Agent(environment),  # pure greedy
        Agent(environment, epsilon=0.1),  # egreedy with epsilon=0.1
        Agent(environment, ucb_param=2),  # ucb with c=2
        Agent(environment, stepsize=0.1, gradient=True)  # gradient with alpha=0.1
    ]
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)
    title = "Performance of Various Bandit Algorithms in a Non-Stationary Environment"
    graph_results([], [optimal_pulls], title, agents, "figures/bandits_comparison_nonstationary.png")
    print()


def compare_envs():
    """Compare stationary and nonstationary environments with 10000 steps and 0.1 epsilon
    """
    stationary_env = Environment(arms=4)
    nonstationary_env = Environment(arms=4, stationary=False, decay=0.05)
    print("Running Experiment...")
    s_reward, s_optimal = run_experiment([Agent(stationary_env, 0.1)], stationary_env, steps=1000)
    n_reward, n_optimal = run_experiment([Agent(nonstationary_env, 0.1)], nonstationary_env, steps=1000)
    print()

    average_reward = [s_reward, n_reward]
    optimal_pulls = [s_optimal, n_optimal]
    title = "eGreedy with ε=0.1 in Stationary and Non-stationary Environments"
    legend = ["Stationary", "Nonstationary"]
    save_loc = "figures/egreedy_environment_comparison.png"

    graph_results(average_reward, optimal_pulls, title, legend, save_loc)


def compare_ns_stepsizes():
    """Compare ERWA and sample average stepsize for non-stationary environment with 10000 steps and 0.1 epsilon
    """
    env = Environment(arms=4, stationary=False, decay=0.05)
    print("Running Experiment...")
    sa_reward, sa_optimal = run_experiment([Agent(env, 0.1, None)], env, steps=10000)
    env.reset()
    erwa_reward, erwa_optimal = run_experiment([Agent(env, 0.1, 0.1)], env, steps=10000)
    print()

    average_reward = [sa_reward, erwa_reward]
    optimal_pulls = [sa_optimal, erwa_optimal]
    title = "Sample Averaging vs Constant Stepsize in a Non-stationary Environment"
    legend = ["stepsize = 1/n", "stepsize = 0.1"]
    save_loc = "figures/nonstationary_stepsize_comparison.png"

    graph_results(average_reward, optimal_pulls, title, legend, save_loc)


def main():
    # compare_stationary_eps()
    # compare_envs()
    # compare_stationary_ucb()
    # compare_ns_stepsizes()
    compare_stationary_gradients()


if __name__ == "__main__":
    main()
