import numpy as np
import matplotlib.pyplot as plt


class TestBed:

    def __init__(self, mean=0, stdev=1, arms=10, runs=2000) -> None:
        """10 different arms (options) with each option initialized from a normal/Gaussian distribution

        Args:
            mean (int, optional): _description_. Defaults to 0.
            stdev (int, optional): _description_. Defaults to 1.
            arms (int, optional): _description_. Defaults to 10.
            runs (int, optional): how many bandit problems. Defaults to 2000.
        """
        self.mean = mean
        self.stdev = stdev
        self.arms = arms
        self.runs = runs
        self.bandits = []

        self.reset()

    def reset(self) -> None:
        # generate q*(a) for each action value
        self.means = np.random.normal(self.mean, self.stdev, self.arms)

        # optimal action
        self.opt = np.argmax(self.means)

    def show_plot(self) -> None:
        """Generate a violin plot of the reward distributions for each action value, like Figure 2.1 in the textbook.
        """
        for i in range(self.arms):
            bandit = np.random.normal(self.means[i], self.stdev, self.runs)
            self.bandits.append(bandit)

        # violin plot with means and stdevs
        fig, ax = plt.subplots()
        ax.violinplot(dataset=self.bandits, showmeans=True, showextrema=False)

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


# testbed = TestBed()
# testbed.show_plot()
