from environment import Environment
from agent import Agent

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


class UCBAgent:

    def __init__(self, arms, ucb_param, trials):
        self.arms = arms
        self.c = ucb_param
        self.trials = trials

        self.estimates = np.zeros(self.arms)  # estimated q
        self.times_taken = np.zeros(self.arms)  # n
        self.action_prob = np.zeros(self.arms)  # likelihood array
        self.likelihoods = np.zeros(self.trials)  # array of likelihoods

    def __str__(self) -> str:
        return f"UCB c = {self.c}"

    def update_estimates(self, time, reward, choice):
        ucb_estimate = self.estimates + (self.c * np.sqrt(np.log(time + 1) / (self.times_taken + 1e-5)))

        if not np.all(ucb_estimate == 0):
            self.action_prob = ucb_estimate / np.sum(ucb_estimate)
        else:
            self.action_prob = np.ones(self.arms) / self.arms

        if np.isnan(choice):
            self.likelihoods[time] = 1  # log(1) = 0
            return

        choice = int(choice)
        self.likelihoods[time] = self.action_prob[choice - 1]

        # we do choice-1 because the data is 1-indexed
        self.times_taken[choice - 1] += 1  # update n

        # # sample average stepsize: stepsize = 1/n
        # self.estimates[choice - 1] += (reward - self.estimates[choice - 1]) / self.times_taken[choice - 1]

        # constant stepsize
        step_size = .9  # chosen at random
        self.estimates[choice - 1] += step_size * (reward - self.estimates[choice - 1])

    def reset(self):
        self.estimates = np.zeros(self.arms)
        self.times_taken = np.zeros(self.arms)
        self.action_prob = np.zeros(self.arms)
        self.likelihoods = np.zeros(self.trials)


def load_data(data):
    df = pd.read_csv(data)
    time = np.array(df['Trial'])
    rewards = np.array(df['Reward'])
    choices = np.array(df['Choice'])
    return time, rewards, choices  # can divide rewards by 100... why do so?


def negative_log_likelihood(ucb_param, trials, data):
    time, rewards, choices = load_data(data)
    agent = UCBAgent(arms=4, ucb_param=ucb_param, trials=trials)
    for t in time:
        agent.update_estimates(t - 1, rewards[t - 1], choices[t - 1])
    log_likelihood = -(np.sum(np.log(agent.likelihoods)))
    # agent.reset()
    return log_likelihood


def manual_optimization(data, trials):
    params = np.arange(0.01, 5, 0.01)
    param_list = []  # x axis
    likelihood_list = []  # y axis

    for param in params:
        log_likelihood = negative_log_likelihood(param, trials, data)
        param_list.append(param)
        likelihood_list.append(log_likelihood)

    # find param for min log likelihood
    min_likelihood = np.min(likelihood_list)
    min_param = param_list[np.argmin(likelihood_list)]

    print("MANUAL OPTIMIZATION ====================")
    print(f"Optimized param c = {min_param}")
    print(f"Log likelihood = {min_likelihood}")

    return min_likelihood, min_param, param_list, likelihood_list


def scipy_optimization(data, trials):
    param_bounds = (0, 5)
    res = minimize_scalar(negative_log_likelihood, bounds=param_bounds, args=(trials, data,))
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


def initialize_arms(data):
    df = pd.read_csv(data)
    arms = []
    num_arms = len(df.columns)
    for i in range(num_arms):
        arm = np.array(df[f'Arm_{i + 1}'])
        # arm = np.array(df[f'Arm_{i + 1}'] / 100)
        arms.append(arm)
    return arms


def run_bandit(arms, runs, ucb_param, arm_vals):
    environment = Environment(arms=arms, runs=runs)
    ucb_agent = Agent(environment=environment, ucb_param=ucb_param)
    average_reward = 0
    for run in range(environment.runs):
        action = ucb_agent.choose_action()
        reward = arm_vals[action][run]
        ucb_agent.update_estimates(reward, action)
        average_reward += reward
    average_reward /= environment.runs
    return average_reward


def main():
    num_participants = 5
    for i in range(num_participants):
        print(f"PARTICIPANT {i + 1}")

        participant = f'P00{i + 1}'
        arm_data = f'data/{participant}_ArmValues.csv'
        behavioural_data = f'data/{participant}_Behavioural.csv'
        arm_vals = initialize_arms(arm_data)
        num_arms = len(arm_vals)
        trials = len(arm_vals[0])

        min_likelihood1, min_param1, param_list, likelihood_list = manual_optimization(behavioural_data, trials)
        min_param2, min_likelihood2 = scipy_optimization(behavioural_data, trials)
        # plot_graph(param_list, likelihood_list, min_param1, min_likelihood1, min_param2, min_likelihood2)

        average_reward = run_bandit(num_arms, trials, min_param1, arm_vals)  # using manual optimization output

        df = pd.read_csv(behavioural_data)
        # df['Reward'] = df['Reward'] / 100
        behavioural_average_reward = df['Reward'].sum() / trials

        print("========================================")
        print(f"Average reward for UCB agent = {average_reward}")
        print(f"Average reward for participant data = {behavioural_average_reward}")
        print()


if __name__ == "__main__":
    main()
