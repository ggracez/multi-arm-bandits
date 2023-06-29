from environment import Environment
from agent import Agent

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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
    """Graph results: average reward and % optimal action

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
        # Average Reward
        for line in average_reward:
            ax1.plot(line)
        ax1.set_ylabel("Average Reward")
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Steps")
        ax1.legend(legend)

    if optimal_pulls:
        # % Optimal Action
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
    # fig.savefig(save_loc)


def compare_stationary_eps():
    """Compare different epsilon values in a stationary environment
    """
    environment = Environment()  # can change # of runs here (default 2000)
    agents = []
    epsilon_vals = [0.1, 0.01, 0]
    for val in epsilon_vals:
        agents.append(Agent(environment, policy="egreedy", param=val))
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)  # can change # of steps here (default 1000)
    title = "Different Epsilon Values in a Stationary Environment"
    graph_results([average_reward], [optimal_pulls], title, agents, "figures/2.2_comparison.png")
    print()


def compare_stationary_ucb():
    """Compare eGreedy and UCB in a stationary environment
    """
    environment = Environment()
    egreedy_agent = Agent(environment, policy="egreedy", param=0.1)
    ucb_agent = Agent(environment, policy="ucb", param=2)
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment([egreedy_agent, ucb_agent], environment)
    title = "eGreedy with ε=0.1 and UCB with c=2 in a Stationary Environment"
    legend = ["eGreedy", "UCB"]
    save_loc = "figures/2.4_egreedy_ucb_comparison.png"
    graph_results([average_reward], [optimal_pulls], title, legend, save_loc)
    print()


def compare_stationary_gradients():
    """Compare different gradient values in a stationary environment
    """
    environment = Environment(mean=4)  # shift the mean to see effects of the baseline
    agents = []
    alpha_vals = [0.1, 0.4]
    for val in alpha_vals:
        agents.append(Agent(environment, policy="gradient", param=val))
        agents.append(Agent(environment, policy="gradient", param=val, baseline=True))
    # run the experiment!!
    print("Running Experiment...")
    average_reward, optimal_pulls = run_experiment(agents, environment)
    title = "Performance of Gradient Bandit Algorithms in a Stationary Environment"
    graph_results([], [optimal_pulls], title, agents, "figures/2.5_gradient_comparison.png")
    print()


def compare_stationary_all():
    """Comparison between eGreedy, UCB, and gradient bandit algorithms in a stationary environment
    """
    environment = Environment()
    agents = [
        Agent(environment, policy="egreedy", param=0),  # pure greedy
        Agent(environment, policy="egreedy", param=0.1),  # egreedy with ε=0.1
        Agent(environment, policy="egreedy", param=0.1, stepsize=0.1),  # egreedy with constant stepsize
        Agent(environment, policy="ucb", param=2),  # ucb with c=2
        Agent(environment, policy="gradient", param=0.1),  # gradient w α=0.1, no need for baseline bc stationary mean=0
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
        Agent(environment, policy="egreedy", param=0),  # pure greedy
        Agent(environment, policy="egreedy", param=0.1),  # egreedy with epsilon=0.1
        Agent(environment, policy="egreedy", param=0.1, stepsize=0.1),  # egreedy with constant stepsize
        Agent(environment, policy="ucb", param=2),  # ucb with c=2
        Agent(environment, policy="gradient", param=0.1, baseline=True)  # gradient with baseline
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
    s_reward, s_optimal = run_experiment([Agent(stationary_env, "egreedy", 0.1)], stationary_env, steps=1000)
    n_reward, n_optimal = run_experiment([Agent(nonstationary_env, "egreedy", 0.1)], nonstationary_env, steps=1000)
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
    sa_reward, sa_optimal = run_experiment([Agent(env, "egreedy", param=0.1)], env, steps=10000)
    env.reset()
    erwa_reward, erwa_optimal = run_experiment([Agent(env, "egreedy", param=0.1, stepsize=0.1)], env, steps=10000)
    print()

    average_reward = [sa_reward, erwa_reward]
    optimal_pulls = [sa_optimal, erwa_optimal]
    title = "Sample Averaging vs Constant Stepsize in a Non-stationary Environment"
    legend = ["stepsize = 1/n", "stepsize = 0.1"]
    save_loc = "figures/nonstationary_stepsize_comparison.png"

    graph_results(average_reward, optimal_pulls, title, legend, save_loc)


# compare_stationary_eps()
# compare_envs()
# compare_stationary_ucb()
# compare_ns_stepsizes()
# compare_stationary_gradients()
# compare_stationary_all()
# compare_nonstationary_all()

