from environment import Environment
from agent import Agent

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar


class BaseAgent:
    def __init__(self, arms, param, trials, decay, stepsize=.9):
        self.arms = arms
        self.epsilon = self.c = self.alpha = self.temp = param
        self.trials = trials
        self.stepsize = stepsize  # learning rate
        self.decay = decay

        self.estimates = np.zeros(self.arms)  # estimated q
        self.times_taken = np.zeros(self.arms)  # n
        self.action_prob = np.zeros(self.arms)  # likelihood array
        self.likelihoods = np.zeros(self.trials)  # array of likelihoods

    def update_estimates(self, time, reward, choice):
        choice = int(choice)
        self.likelihoods[time] = self.action_prob[choice - 1]

        # we do choice-1 because the data is 1-indexed
        self.times_taken[choice - 1] += 1  # update n

        if self.decay:
            for arm in range(self.arms):
                if arm != choice - 1:
                    self.estimates[arm] *= self.decay
                else:
                    self.estimates[arm] = self.estimates[arm] * self.decay + reward
        else:
            # # sample average stepsize: stepsize = 1/n
            # self.estimates[choice - 1] += (reward - self.estimates[choice - 1]) / self.times_taken[choice - 1]

            # constant stepsize
            # step_size = .9  # chosen at random
            self.estimates[choice - 1] += self.stepsize * (reward - self.estimates[choice - 1])

    def reset(self):
        self.estimates = np.zeros(self.arms)  # estimated q
        self.times_taken = np.zeros(self.arms)  # n
        self.action_prob = np.zeros(self.arms)  # likelihood array
        self.likelihoods = np.zeros(self.trials)  # array of likelihoods


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


class SoftmaxAgent(BaseAgent):

    def __str__(self) -> str:
        return f"Softmax τ = {self.temp}"

    def get_likelihoods(self, time, reward, choice):
        if np.isnan(choice):
            self.likelihoods[time] = 1  # log(1) = 0
            return

        exponential = np.exp(self.estimates / self.temp)
        self.action_prob = exponential / np.sum(exponential)

        self.update_estimates(time, reward, choice)


class GradientAgent(BaseAgent):

    def __str__(self) -> str:
        return f"Gradient α = {self.alpha}"

    def get_likelihoods(self, time, reward, choice):
        if np.isnan(choice):
            self.likelihoods[time] = 1  # log(1) = 0
            return

        exponential = np.exp(self.estimates)
        self.action_prob = exponential / np.sum(exponential)

        self.update_estimates(time, reward, choice)

    def update_estimates(self, time, reward, choice):
        choice = int(choice)
        self.likelihoods[time] = self.action_prob[choice - 1]
        self.times_taken[choice - 1] += 1
        one_hot = np.zeros(self.arms)
        one_hot[choice - 1] = 1
        self.estimates += self.alpha * reward * (one_hot - self.action_prob)


def load_data(data):
    df = pd.read_csv(data)
    time = np.array(df['Trial'])
    rewards = np.array(df['Reward'])
    choices = np.array(df['Choice'])
    return time, rewards / 100, choices  # can divide rewards by 100... np errors if not when using softmax


def negative_log_likelihood(param, stepsize, decay, model, trials, data):
    time, rewards, choices = load_data(data)
    if model == "UCB":
        agent = UCBAgent(arms=4, param=param, trials=trials, stepsize=stepsize)
    elif model == "Gradient":
        agent = GradientAgent(arms=4, param=param, trials=trials, stepsize=stepsize)
    elif model == "Softmax":
        agent = SoftmaxAgent(arms=4, param=param, trials=trials, stepsize=stepsize)
    else:  # model is eGreedy
        agent = eGreedyAgent(arms=4, param=param, trials=trials, stepsize=stepsize, decay=decay)
    for t in time:
        agent.get_likelihoods(t - 1, rewards[t - 1], choices[t - 1])
    log_likelihood = -(np.sum(np.log(agent.likelihoods)))
    return log_likelihood


def manual_optimization(data, model, trials):
    params = np.arange(0.01, 1, 0.01)
    # stepsizes = np.arange(0.1, 1, 0.1)
    stepsizes = [0.9]
    decays = np.arange(0.1, 1, 0.1)
    param_list = []  # x axis
    stepsize_list = []
    decay_list = []
    likelihood_list = []  # y axis

    for param in params:
        for stepsize in stepsizes:
            for decay in decays:
                log_likelihood = negative_log_likelihood(param, stepsize, decay, model, trials, data)
                param_list.append(param)
                stepsize_list.append(stepsize)
                decay_list.append(decay)
                likelihood_list.append(log_likelihood)

    # find params for min log likelihood
    min_likelihood = np.min(likelihood_list)
    min_decay = decay_list[np.argmin(likelihood_list)]
    min_param = param_list[np.argmin(likelihood_list)]
    min_stepsize = stepsize_list[np.argmin(likelihood_list)]

    print(f"Optimized param (MANUAL) = {min_param}")
    print(f"Optimized stepsize (MANUAL) = {min_stepsize}")
    print(f"Log likelihood (MANUAL) = {min_likelihood}")
    print(f"Optimized decay (MANUAL) = {min_decay}")

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
        # arm = np.array(df[f'Arm_{i + 1}'])
        arm = np.array(df[f'Arm_{i + 1}'] / 100)
        arms.append(arm)
    return arms


def run_bandits(model, arms, runs, param, arm_vals):
    environment = Environment(arms=arms, runs=runs)
    if model == "UCB":
        agent = Agent(environment=environment, policy="ucb", param=param)
    elif model == "Gradient":
        agent = Agent(environment=environment, policy="gradient", param=param)
    else:  # model is eGreedy
        agent = Agent(environment=environment, policy="egreedy", param=param)
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
    # models = ["UCB", "eGreedy", "Gradient"]
    models = ["eGreedy"]
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
            # min_param2 = scipy_optimization(behavioural_data, model, trials)
            average_reward = run_bandits(model, num_arms, trials, min_param1, arm_vals)
            model_ave_rewards[model] = average_reward

        print("=" * 30)

        for model in models:
            print(f"Average reward for {model} agent = {model_ave_rewards[model]}")

        df = pd.read_csv(behavioural_data)
        df['Reward'] = df['Reward'] / 100
        behavioural_average_reward = df['Reward'].sum() / trials
        print(f"Average reward for participant data = {behavioural_average_reward}")
        print()


if __name__ == "__main__":
    main()
