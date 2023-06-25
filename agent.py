import numpy as np


class Agent:

    def __init__(self, environment, epsilon=0.0, stepsize=None, ucb_param=None, gradient=False,
                 baseline=False, baseline_stepsize=None):

        # policy=[egreedy, ucb, gradient]
        # param = [epsilon, ucb_param, stepsize]
        self.environment = environment
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.c = ucb_param
        self.gradient = gradient
        self.baseline = baseline
        self.baseline_stepsize = baseline_stepsize
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
            if self.stepsize:
                return f"ε = {self.epsilon} with stepsize = {self.stepsize}"
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
        self.average_reward += (reward - self.average_reward) / self.time  # TODO: do for nonstationary (gradient)

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
