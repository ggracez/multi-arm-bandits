from environment import Environment
from agent import Agent

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar


class BaseAgent:
    def __init__(self, arms, param, trials):
        self.arms = arms
        self.epsilon = self.c = param
        self.trials = trials

        self.estimates = np.zeros(self.arms)  # estimated q
        self.times_taken = np.zeros(self.arms)  # n
        self.action_prob = np.zeros(self.arms)  # likelihood array
        self.likelihoods = np.zeros(self.trials)  # array of likelihoods

    def update_estimates(self, time, reward, choice):
        choice = int(choice)
        self.likelihoods[time] = self.action_prob[choice - 1]

        # we do choice-1 because the data is 1-indexed
        self.times_taken[choice - 1] += 1  # update n

        # # sample average stepsize: stepsize = 1/n
        # self.estimates[choice - 1] += (reward - self.estimates[choice - 1]) / self.times_taken[choice - 1]

        # constant stepsize
        step_size = .9  # chosen at random
        self.estimates[choice - 1] += step_size * (reward - self.estimates[choice - 1])


class eGreedyAgent(BaseAgent):

    def __str__(self) -> str:
        return f"eGreedy ε = {self.epsilon}"

    def get_likelihoods(self, time, reward, choice):
        if np.isnan(choice):
            self.likelihoods[time] = 1  # log(1) = 0
            return

        greedy_action = np.argmax(self.estimates)
        self.action_prob = np.ones(self.arms) * (self.epsilon / (self.arms - 1))
        self.action_prob[greedy_action] = 1 - self.epsilon

        self.update_estimates(time, reward, choice)


class UCBAgent(BaseAgent):

    def __str__(self) -> str:
        return f"UCB c = {self.c}"

    def get_likelihoods(self, time, reward, choice):
        if np.isnan(choice):
            self.likelihoods[time] = 1  # log(1) = 0
            return

        ucb_estimate = self.estimates + (self.c * np.sqrt(np.log(time + 1) / (self.times_taken + 1e-5)))

        if not np.all(ucb_estimate == 0):
            self.action_prob = ucb_estimate / np.sum(ucb_estimate)
        else:
            self.action_prob = np.ones(self.arms) / self.arms

        self.update_estimates(time, reward, choice)


def load_data(data):
    df = pd.read_csv(data)
    time = np.array(df['Trial'])
    rewards = np.array(df['Reward'])
    choices = np.array(df['Choice'])
    return time, rewards, choices  # can divide rewards by 100... why do so?


def negative_log_likelihood(param, model, trials, data):
    time, rewards, choices = load_data(data)
    if model == "UCB":
        agent = UCBAgent(arms=4, param=param, trials=trials)
    else:  # model is eGreedy
        agent = eGreedyAgent(arms=4, param=param, trials=trials)
    for t in time:
        agent.get_likelihoods(t - 1, rewards[t - 1], choices[t - 1])
    log_likelihood = -(np.sum(np.log(agent.likelihoods)))
    return log_likelihood


def manual_optimization(data, model, trials):
    params = np.arange(0.01, 1, 0.01)
    param_list = []  # x axis
    likelihood_list = []  # y axis

    for param in params:
        log_likelihood = negative_log_likelihood(param, model, trials, data)
        param_list.append(param)
        likelihood_list.append(log_likelihood)

    # find param for min log likelihood
    min_likelihood = np.min(likelihood_list)
    min_param = param_list[np.argmin(likelihood_list)]

    print(f"Optimized param (MANUAL) = {min_param}")
    print(f"Log likelihood (MANUAL) = {min_likelihood}")

    return min_param


def scipy_optimization(data, model, trials):
    param_bounds = (0, 1)
    res = minimize_scalar(negative_log_likelihood, bounds=param_bounds, args=(model, trials, data,))
    print(f"Optimized param (SCIPY) = {res.x}")
    print(f"Log likelihood (SCIPY) = {res.fun}")

    return res.x


def initialize_arms(data):
    df = pd.read_csv(data)
    arms = []
    num_arms = len(df.columns)
    for i in range(num_arms):
        arm = np.array(df[f'Arm_{i + 1}'])
        # arm = np.array(df[f'Arm_{i + 1}'] / 100)
        arms.append(arm)
    return arms


def run_bandits(model, arms, runs, param, arm_vals):
    environment = Environment(arms=arms, runs=runs)
    if model == "UCB":
        agent = Agent(environment=environment, ucb_param=param)
    else:  # model is eGreedy
        agent = Agent(environment=environment, epsilon=param)
    average_reward = 0
    for run in range(environment.runs):
        action = agent.choose_action()
        reward = arm_vals[action][run]
        agent.update_estimates(reward, action)
        average_reward += reward
    average_reward /= environment.runs
    return average_reward


def main():
    num_participants = 5
    models = ["UCB", "eGreedy"]
    model_ave_rewards = {}
    for i in range(num_participants):
        print(f"PARTICIPANT {i + 1}")

        participant = f'P00{i + 1}'
        arm_data = f'data/{participant}_ArmValues.csv'
        behavioural_data = f'data/{participant}_Behavioural.csv'
        arm_vals = initialize_arms(arm_data)
        num_arms = len(arm_vals)
        trials = len(arm_vals[0])

        for model in models:
            num_spaces = 30 - len(model)
            divider = "=" * num_spaces
            print(f"MODEL: {model} {divider}")
            min_param1 = manual_optimization(behavioural_data, model, trials)
            min_param2 = scipy_optimization(behavioural_data, model, trials)
            average_reward = run_bandits(model, num_arms, trials, min_param1, arm_vals)
            model_ave_rewards[model] = average_reward

        print("=" * 30)

        for model in models:
            print(f"Average reward for {model} agent = {model_ave_rewards[model]}")

        df = pd.read_csv(behavioural_data)
        # df['Reward'] = df['Reward'] / 100
        behavioural_average_reward = df['Reward'].sum() / trials
        print(f"Average reward for participant data = {behavioural_average_reward}")
        print()


if __name__ == "__main__":
    main()
