import numpy as np
from environment import Environment


class Agent:

    def __init__(self, environment, policy, param, stepsize=None, baseline=False, explore=None, decay=None):
        """
        Args:
            environment (Environment)
            policy (str): eGreedy, UCB, or gradient
            param (float): epsilon, c, alpha, or tau
            stepsize (float, optional): learning rate parameter. Defaults to None.
            baseline (bool, optional): Whether to use a baseline. Defaults to False.
            explore (float, optional): Explore parameter for softmax. Defaults to None.
            decay (float, optional): Decay parameter between 0 and 1. Defaults to None.
        """

        self.environment = environment
        self.policy = policy.lower()

        if self.policy == "egreedy":
            self.epsilon = param

        elif self.policy == "ucb":
            self.c = param

        elif self.policy == "gradient":
            self.alpha = param
            self.baseline = baseline
            self.average_reward = 0

        elif self.policy == "softmax":
            self.temp = param
            self.explore = explore

        self.stepsize = stepsize  # for gradient, this is the baseline stepsize (not implemented)
        self.decay = decay

        self.estimates = np.zeros(self.environment.arms)  # estimated q
        self.times_taken = np.zeros(self.environment.arms)  # n
        self.action_prob = np.zeros(self.environment.arms)  # probability π
        self.action_history = np.zeros(self.environment.arms)  # history of actions taken (T_j for softmax explore)
        self.time = 0  # t

    def __str__(self) -> str:

        if self.policy == "egreedy":
            if self.epsilon == 0:
                return "ε = 0 (greedy)"
            s = f"ε = {self.epsilon}"
            if self.stepsize:
                if self.decay:
                    s += f" with stepsize = {self.stepsize} and decay = {self.decay}"
                else:
                    s += f" with stepsize = {self.stepsize}"
            elif self.decay:
                s += f" with decay = {self.decay}"
            return s

        # if self.policy == "egreedy":
        #     if self.epsilon == 0:
        #         return f"ε = 0 (greedy)"
        #     elif self.stepsize:
        #         return f"ε = {self.epsilon} with stepsize = {self.stepsize}"
        #     else:
        #         return f"ε = {self.epsilon}"

        elif self.policy == "ucb":
            return f"UCB c = {self.c}"

        elif self.policy == "gradient":
            if self.baseline:
                return f"Gradient α = {self.alpha} with baseline"
            else:
                return f"Gradient α = {self.alpha}"

        elif self.policy == "softmax":
            if self.explore:
                return f"Softmax τ = {self.temp} with explore = {self.explore}"
            elif self.decay:  # later may need decay + explore?
                return f"Softmax τ = {self.temp} with decay = {self.decay}"
            else:
                return f"Softmax τ = {self.temp}"

    def choose_action(self) -> int:
        """
        Returns:
            int: the best action (which arm to pull)
        """
        if self.policy == "egreedy":
            if np.random.random() < self.epsilon:  # explore
                action = np.random.choice(len(self.estimates))  # = np.random.choice(self.environment.arms)
            else:  # exploit
                greedy_action = np.argmax(self.estimates)

                # find actions with same value as greedy action
                action = np.where(self.estimates == greedy_action)[0]  # returns list of actions

                # choose one of them at random (accounts for duplicates)
                if len(action) == 0:
                    action = greedy_action
                else:
                    action = np.random.choice(action)

        elif self.policy == "ucb":
            # get UCB estimate (add 1e-5 to avoid divide by 0)
            ucb_estimate = self.estimates + (self.c * np.sqrt(np.log(self.time + 1) / (self.times_taken + 1e-5)))
            ucb_action = np.argmax(ucb_estimate)
            action = np.where(ucb_estimate == ucb_action)[0]
            if len(action) == 0:
                action = ucb_action
            else:
                action = np.random.choice(action)

        elif self.policy == "gradient":
            exponential = np.exp(self.estimates)
            self.action_prob = exponential / np.sum(exponential)
            action = np.random.choice(len(self.estimates), p=self.action_prob)

        elif self.policy == "softmax":
            # NOTE: do we add 1 to time???? we should be counting the current trial, I forget if the trial is 0 or 1
            if self.explore:
                uncertainty = self.explore * (self.time - self.action_history)
                exponential = np.exp((self.estimates / self.temp) + uncertainty)
            else:
                exponential = np.exp(self.estimates / self.temp)
            self.action_prob = exponential / np.sum(exponential)
            action = np.random.choice(len(self.estimates), p=self.action_prob)

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
        self.action_history[action] = self.time  # update last action MAY NOT NEED

        if self.policy == "gradient":
            # gradient algorithm uses action preferences, updated with:
            #   pref += stepsize * (reward - baseline) * (1 - prob of action) for all selected actions and
            #   pref += stepsize * (reward - baseline) * (0 - prob of action) for all non-selected actions
            # baseline is average reward up to but not including current time step - very good if mean != 0
            #   if reward > baseline, then the probability of taking A_t in the future is increased and vice versa
            #   non-selected actions move in the opposite direction
            if self.baseline:
                self.average_reward += (reward - self.average_reward) / self.time
                baseline = self.average_reward
            else:
                baseline = 0
            one_hot = np.zeros(self.environment.arms)
            one_hot[action] = 1
            self.estimates += self.alpha * (reward - baseline) * (one_hot - self.action_prob)

        else:
            if self.decay:
                for arm in range(self.environment.arms):
                    if arm != action:
                        self.estimates[arm] = self.estimates[arm] * self.decay
                    else:
                        self.estimates[arm] = self.estimates[arm] * self.decay + reward
            else:
                if not self.stepsize:
                    # sample average stepsize: stepsize = 1/n
                    self.estimates[action] += (reward - self.estimates[action]) / self.times_taken[action]
                else:
                    # constant stepsize: exponential recency weighted average (ERWA)
                    self.estimates[action] += self.stepsize * (reward - self.estimates[action])

    def reset(self):
        """Reset to initial values
        """
        self.estimates = np.zeros(self.environment.arms)
        self.times_taken = np.zeros(self.environment.arms)
        self.action_prob = np.zeros(self.environment.arms)
        self.action_history = np.zeros(self.environment.arms)
        self.time = 0


def test():
    print()
    # estimates = [0, 0, 0, 0], action_history = [0, 0, 0, 0]
    agent = Agent(Environment(arms=4), policy="egreedy", param=0.1, decay=0.1)

    # choose action 0, get reward 1
    # action 0: ev = ev * decay + reward = 0 * 0.1 + 1 * 1 = 1
    # estimates = [1, 0, 0, 0]
    print(f"reward = 1, action = 0: {agent.estimates=}")
    agent.update_estimates(reward=1, action=0)

    # choose action 1, get reward 0
    # action 0: ev = ev * decay = 1 * 0.1 = 0.1
    # action 1: ev = ev * decay + reward = 0 * 0.1 + 1 = 1
    # estimates [0.1, 1, 0, 0]
    print(f"reward = 1, action = 1: {agent.estimates=}")
    agent.update_estimates(reward=1, action=1)

    # choose action 3, get reward 1
    # action 0: ev * decay = 0.1 * 0.1 = 0.01
    # action 1: ev * decay = 1 * 0.1 = 0.1
    # action 2: ev * decay = 0 * 0.1 = 0
    # action 3: ev * decay + reward = 0 * 0.1 + 1 = 1
    # estimates [0.01, 0.1, 0, 1]
    print(f"reward = 1, action = 3: {agent.estimates=}")
    agent.update_estimates(reward=1, action=3)

    print(f"reward = 1, action = 3: {agent.estimates=}")
    agent.update_estimates(reward=1, action=3)
