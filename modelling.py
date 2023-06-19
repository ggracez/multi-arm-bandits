import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


class UCBAgent:

    def __init__(self, arms, ucb_param):
        self.arms = arms
        self.c = ucb_param

        self.estimates = np.zeros(self.arms)  # estimated q
        self.times_taken = np.zeros(self.arms)  # n
        self.action_prob = np.zeros(self.arms)  # likelihood array

    def __str__(self) -> str:
        return f"UCB c = {self.c}"

    def update_estimates(self, time, reward, choice):
        ucb_estimate = self.estimates + (self.c * np.sqrt(np.log(time + 1) / (self.times_taken + 1e-5)))

        if not np.all(ucb_estimate == 0):
            self.action_prob = ucb_estimate / np.sum(ucb_estimate)
        else:
            self.action_prob = np.ones(self.arms) / self.arms

        # we do choice-1 because the data is 1-indexed
        self.times_taken[choice - 1] += 1  # update n

        # sample average stepsize: stepsize = 1/n
        self.estimates[choice - 1] += (reward - self.estimates[choice - 1]) / self.times_taken[choice - 1]  # update q

    def reset(self):
        self.estimates = np.zeros(self.arms)
        self.times_taken = np.zeros(self.arms)
        self.action_prob = np.zeros(self.arms)


def load_data(data):
    df = pd.read_csv(data)
    time = np.array(df['Trial']).astype(int)
    rewards = np.array(df['Reward']).astype(int)
    choices = np.array(df['Choice']).astype(int)
    return time, rewards, choices


def negative_log_likelihood(param, data):
    time, rewards, choices = load_data(data)
    agent = UCBAgent(arms=4, ucb_param=param)
    for t in time:
        agent.update_estimates(t - 1, rewards[t - 1], choices[t - 1])
    log_likelihood = -(np.sum(np.log(agent.action_prob)))
    agent.reset()
    return log_likelihood


def manual_optimization(data):
    params = np.arange(0.5, 300, 0.5)
    # res = {}
    # param_list = np.array([])  # x axis
    # likelihood_list = np.array([])  # y axis
    param_list = []  # x axis
    likelihood_list = []  # y axis

    for param in params:
        log_likelihood = negative_log_likelihood(param, data)
        # res[param] = log_likelihood
        # param_list = np.append(param_list, param)
        # likelihood_list = np.append(likelihood_list, log_likelihood)
        param_list.append(param)
        likelihood_list.append(log_likelihood)

    # find param for min log likelihood
    min_likelihood = np.min(likelihood_list)
    min_param = param_list[np.argmin(likelihood_list)]

    print("MANUAL OPTIMIZATION ====================")
    print(f"Optimized param c = {min_param}")
    print(f"Log likelihood = {min_likelihood}")

    return min_likelihood, min_param, param_list, likelihood_list


def scipy_optimization(data):
    param_bounds = (0, 300)
    res = minimize_scalar(negative_log_likelihood, bounds=param_bounds, args=(data,))
    print("SCIPY OPTIMIZATION =====================")
    print(f"Optimized param c = {res.x}")
    print(f"Log likelihood = {res.fun}")

    return res.x, res.fun


def plot_graph(param_list, likelihood_list, min_param1, min_likelihood1, min_param2, min_likelihood2):
    plt.plot(param_list, likelihood_list)
    plt.xlabel('UCB Parameter')
    plt.ylabel('Log Likelihood')
    plt.title('UCB Parameter vs Log Likelihood')

    # add marker to display min for manual optimization
    plt.scatter(min_param1, min_likelihood1, marker='o', color='orange')
    y_offset = 0.05 * (np.max(likelihood_list) - np.min(likelihood_list))
    plt.text(min_param1, min_likelihood1 + y_offset, f'c = {min_param1:.1f}', color='orange')

    # add marker to display min for scipy optimization
    plt.scatter(min_param2, min_likelihood2, marker='o', color='purple')
    y_offset = 0.05 * (np.max(likelihood_list) - np.min(likelihood_list))
    plt.text(min_param2, min_likelihood2 + y_offset, f'c = {min_param2:.1f}', color='purple')

    plt.show()


def main():
    data = 'data/4Arm_Bandit_Behavioural.csv'
    min_likelihood1, min_param1, param_list, likelihood_list = manual_optimization(data)
    min_param2, min_likelihood2 = scipy_optimization(data)
    # plot_graph(param_list, likelihood_list, min_param1, min_likelihood1, min_param2, min_likelihood2)


if __name__ == "__main__":
    main()
