import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from environment import Environment
from agent import Agent
from model_fitting import eGreedyAgent, UCBAgent, GradientAgent


def initialize_arms(arm_data):
    df = pd.read_csv(arm_data)
    arms = []
    for i in range(len(df.columns)):
        arm = np.array(df[f'Arm_{i + 1}'])
        arms.append(arm)
    return arms


def generate_simulated(model, param, stepsize, arms):
    num_arms = len(arms)
    runs = len(arms[0])
    choice_array = np.zeros(runs)
    reward_array = np.zeros(runs)
    agent = Agent(environment=Environment(arms=num_arms), policy=model.lower(), param=param, stepsize=stepsize)
    for run in range(runs):
        action = agent.choose_action()
        choice_array[run] = action
        reward = arms[action][run]
        reward_array[run] = reward
        agent.update_estimates(reward, action)
    return choice_array.astype(int), reward_array


def negative_log_likelihood(param, stepsize, model, trials, choices, rewards):
    if model == "eGreedy":
        agent = eGreedyAgent(arms=4, param=param, trials=trials, stepsize=stepsize)
    elif model == "UCB":
        agent = UCBAgent(arms=4, param=param, trials=trials, stepsize=stepsize)
    else:  # model is Gradient
        agent = GradientAgent(arms=4, param=param, trials=trials, stepsize=stepsize)
    for t in range(trials):
        agent.get_likelihoods(t, rewards[t], choices[t])
    log_likelihood = -(np.sum(np.log(agent.likelihoods)))
    return log_likelihood


def scipy_optimization(choices, rewards, model, trials):
    param_bounds = (0, 1)
    res = minimize_scalar(negative_log_likelihood, bounds=param_bounds, args=(model, trials, choices, rewards,))
    print(f"Optimized param (SCIPY) = {res.x}")
    return res.x


def manual_optimization(choices, rewards, model, trials):
    params = np.arange(0.01, 1, 0.01)
    stepsizes = np.arange(0.1, 1, 0.1)
    param_list = []  # x axis
    stepsize_list = []
    likelihood_list = []  # y axis

    for param in params:
        for stepsize in stepsizes:
            log_likelihood = negative_log_likelihood(param, stepsize, model, trials, choices, rewards)
            param_list.append(param)
            stepsize_list.append(stepsize)
            likelihood_list.append(log_likelihood)

    # find params for min log likelihood
    min_param = param_list[np.argmin(likelihood_list)]
    min_stepsize = stepsize_list[np.argmin(likelihood_list)]
    print(f"Optimized param (MANUAL) = {min_param}")
    print(f"Optimized stepsize (MANUAL) = {min_stepsize}")

    return min_param, min_stepsize


def plot_correlation(model, param_type, simulated, recovered):
    fig, ax = plt.subplots(1)
    fig.suptitle(f"{model} Parameter Recovery", fontsize=16)
    ax.set_xlabel(f"Simulated {param_type}")
    ax.set_ylabel(f"Recovered {param_type}")

    ax.scatter(simulated, recovered)

    corr = np.corrcoef(simulated, recovered)[0, 1]
    plt.text(0.05, 0.9, f'r={corr:.3f}', transform=ax.transAxes)
    print('Pearson correlation: %.3f' % corr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", color='gray')

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    plt.show()


def main():
    arm_data_file = "data/P001_ArmValues.csv"
    model = "eGreedy"
    arms = initialize_arms(arm_data_file)
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    recovered_eps = []
    recovered_steps = []
    for e in eps:
        print(f"Simulated epsilon = {e}")
        choices, rewards = generate_simulated(model, e, e, arms)
        # sim = scipy_optimization(choices, rewards, model, len(arms[0]))
        sim_param, sim_step = manual_optimization(choices, rewards, model, len(arms[0]))
        recovered_eps.append(sim_param)
        recovered_steps.append(sim_step)
    plot_correlation(model, "Epsilon", eps, recovered_eps)
    plot_correlation(model, "Learning Rate", eps, recovered_steps)


if __name__ == "__main__":
    main()
