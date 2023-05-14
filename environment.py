import numpy as np
import matplotlib.pyplot as plt


class Environment:

    def __init__(self, mean=0, stdev=1, arms=10, runs=2000, stationary=True, decay=1):
        """Default 10 different arms (options) with each option initialized from a normal/Gaussian distribution.
        Runs is the number of bandit problems.
        """
        self.mean = mean
        self.stdev = stdev
        self.arms = arms
        self.runs = runs
        self.stationary = stationary
        self.decay = decay

        self.reset()

    def reset(self):
        # generate q*(a) for each action value
        self.means = np.random.normal(self.mean, self.stdev, self.arms)

        if not self.stationary:
            # non-stationary environment: random walk arms
            self.update_arms()

        # optimal action
        self.opt = np.argmax(self.means)

    def update_arms(self):
        """Random walk for non-stationary arms
        """
        if not self.stationary:
            for i in range(self.arms):
                walk_size = np.random.normal(self.mean, self.stdev)
                self.means[i] += (walk_size * self.decay)  # higher decay means arms change more rapidly

    def plot_walk(self, trials):
        """Plot changing means for non-stationary arms
        Different arms have different means
        Note that if the arms are stationary, the arms will be a horizontal line
        """
        true_rewards = np.zeros((trials, self.arms))
        for trial in range(trials):
            self.update_arms()
            for i in range(self.arms):
                true_rewards[trial, i] += self.means[i]
        fig, ax = plt.subplots()
        ax.plot(true_rewards)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Mean")
        ax.legend([f"Arm {i+1}" for i in range(self.arms)])
        # fig.savefig("figures/nonstationary.png")
        plt.show()

    def show_plot(self):
        """Generate a violin plot of the reward distributions for each action value, like Figure 2.1 in the textbook.
        """
        bandits = []

        for i in range(self.arms):
            bandit = np.random.normal(self.means[i], self.stdev, self.runs)
            bandits.append(bandit)

        # violin plot with means and stdevs
        fig, ax = plt.subplots()
        ax.violinplot(dataset=bandits, showmeans=True, showextrema=False)

        # x-axis ticks for each action value
        arms = np.arange(1, self.arms + 1)
        ax.set_xticks(arms)

        # # scatterplot of means only
        # ax.scatter(arms, self.means)

        # set axis labels
        ax.set_ylabel("Reward Distribution")
        ax.set_xlabel("Action")

        # center y-axis at y=0
        ax.axhline(y=0, linestyle="--", color="gray")
        y_max = np.abs(ax.get_ylim()).max()
        ax.set_ylim(ymin=-y_max, ymax=y_max)

        plt.show()
        # fig.savefig("figures/2.1_scatter.png")
        # fig.savefig("figures/2.1_violin.png")
        # plt.close()


# # plot fig 2.1
# environment = Environment()
# environment.show_plot()
# environment = Environment(arms=4, stationary=False, decay=0.05)
# environment.plot_walk(300)
