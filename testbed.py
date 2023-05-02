import numpy as np
import matplotlib.pyplot as plt

class TestBed():
    
    def __init__(self, mean=0, stdev=1, arms=10) -> None:
        """10 different arms (options) with each option initialized from a normal/Gaussian distribution

        Args:
            mean (int, optional): _description_. Defaults to 0.
            stdev (int, optional): _description_. Defaults to 1.
            arms (int, optional): _description_. Defaults to 10.
        """
        self.mean = mean
        self.stdev = stdev
        self.arms = arms
        self.bandits = []
        
        # generate q*(a) for each action value
        means = np.random.normal(self.mean, self.stdev, self.arms)
        
        for i in range(self.arms):
            bandit = np.random.normal(means[i], self.stdev, 2000)  # 2000 iterations?
            self.bandits.append(bandit)
    
    def show_plot(self) -> None:

        # violin plot with means and stdevs
        fig, ax = plt.subplots()  # idk what the point of subplots are tbh
        ax.violinplot(dataset=self.bandits, showmeans=True)

        # x-axis ticks for each action value
        arms = np.arange(1, self.arms+1)
        ax.set_xticks(arms)

        # # scatterplot of means only
        # ax.scatter(arms, means)
        
        # set axis labels
        ax.set_ylabel("Reward Distribution")
        ax.set_xlabel("Action")

        # center y-axis at y=0
        ax.axhline(y=0, linestyle="--", color="gray")
        y_max = np.abs(ax.get_ylim()).max()
        ax.set_ylim(ymin=-y_max, ymax=y_max)

        plt.show()
        # fig.savefig("violin.png")  # or plt.savefig ????

testbed = TestBed()
testbed.show_plot()