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


def generate_simulated(model, param, arms):
    num_arms = len(arms)
    runs = len(arms[0])
    choice_array = np.zeros(runs)
    reward_array = np.zeros(runs)
    agent = Agent(environment=Environment(arms=num_arms), policy=model.lower(), param=param)
    for run in range(runs):
        action = agent.choose_action()
        choice_array[run] = action
        reward = arms[action][run]
        reward_array[run] = reward
        agent.update_estimates(reward, action)
    return choice_array.astype(int), reward_array


def negative_log_likelihood(param, model, trials, choices, rewards):
    if model == "eGreedy":
        agent = eGreedyAgent(arms=4, param=param, trials=trials)
    elif model == "UCB":
        agent = UCBAgent(arms=4, param=param, trials=trials)
    else:  # model is Gradient
        agent = GradientAgent(arms=4, param=param, trials=trials)
    for t in range(trials):
        agent.get_likelihoods(t, rewards[t], choices[t])
    log_likelihood = -(np.sum(np.log(agent.likelihoods)))
    return log_likelihood


def scipy_optimization(choices, rewards, model, trials):
    param_bounds = (0, 1)
    res = minimize_scalar(negative_log_likelihood, bounds=param_bounds, args=(model, trials, choices, rewards,))
    print(f"Optimized param (SCIPY) = {res.x}")
    return res.x


def plot_correlation(model, simulated, recovered):
    fig, ax = plt.subplots(1)
    fig.suptitle(f"{model} Parameter Recovery", fontsize=16)
    ax.set_xlabel("Simulated Epsilon")
    ax.set_ylabel("Recovered Epsilon")

    ax.scatter(simulated, recovered)

    corr = np.corrcoef(simulated, recovered)[0, 1]
    plt.text(0.05, 0.9, f'r={corr:.3f}', transform=ax.transAxes)
    print('Pearson correlation: %.3f' % corr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", color='gray')

    plt.show()


def main():
    arm_data_file = "data/P001_ArmValues.csv"
    model = "eGreedy"
    arms = initialize_arms(arm_data_file)
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    recovered_eps = []
    for e in eps:
        print(f"Simulated epsilon = {e}")
        choices, rewards = generate_simulated(model, e, arms)
        sim = scipy_optimization(choices, rewards, model, len(arms[0]))
        recovered_eps.append(sim)
    plot_correlation(model, eps, recovered_eps)


if __name__ == "__main__":
    main()
