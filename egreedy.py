from testbed import TestBed

import numpy as np
import matplotlib.pyplot as plt

class eGreedy():
    
    def __init__(self, testbed, epsilon=0) -> None:
        # q_t(a) = (sum of rewards when a taken prior to t) / (# of times a taken prior to t)
        # q_(n+1) = q_n + 1/n * (r_n - q_n)
        self.testbed = testbed
        self.epsilon = epsilon
        self.timestep = 0  # t
        self.sum_rewards = np.zeros(self.testbed.arms)  # sum of rewards when a taken prior to t
        self.estimates = np.zeros(self.testbed.arms)  # estimated q
        self.times_taken = np.zeros(self.testbed.arms)  # n
    
    def __str__(self) -> str:
        if self.epsilon == 0:
            return f"ε = 0 (greedy)"
        else:
            return f"ε = {self.epsilon}"
    
    def choose_action(self) -> int:
        """either greedy (exploit) or epsilon (explore)

        Returns:
            int: the best action (which arm to pull)
        """
        if np.random.random() < self.epsilon:  # explore
            action = np.random.choice(len(self.estimates))  # = np.random.choice(self.testbed.arms)
        
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
    
    def update_estimates(self, reward:float, action) -> None:
        """update q at time t: 
            q_t(a) = (sum of rewards when a taken prior to t) / (# of times a taken prior to t)

        Args:
            reward (float): reward at time t
            action (int): action at time t
        """
        self.times_taken[action] += 1
        self.sum_rewards[action] += reward
        
        self.estimates[action] = self.sum_rewards[action] / self.times_taken[action]
        
        self.timestep += 1
            
    def reset(self) -> None:
        """Reset to initial values
        """
        self.timestep = 0
        self.sum_rewards = np.zeros(self.testbed.arms)
        self.estimates = np.zeros(self.testbed.arms)
        self.times_taken = np.zeros(self.testbed.arms)


# actually run the bandit problems
def run(agents:list[eGreedy], testbed, steps=1000) -> None:
    """each agent represents a different epsilon value

    Args:
        agents (list[eGreedy]): eGreedy agents with different epsilon values
        testbed (TestBed): defaults to 10-arm testbed with 2000 runs
        steps (int, optional): _description_. Defaults to 1000.
    """
    average_reward = np.zeros((steps, len(agents)))
    optimal_pulls = np.zeros((steps, len(agents)))
    
    # optimal_pulls = []
    # average_reward = []
    
    for run in range(testbed.runs):
        
        if run % 100 == 0:
            print("Run:", run)

        testbed.reset()
        for agent in agents:
            agent.reset()
        
        for step in range(steps):
            agent_num = 0
            for agent in agents:
                action = agent.choose_action()
                reward = np.random.normal(testbed.means[action], 1)
                agent.update_estimates(reward, action)
                
                # arr[row, col] = arr[row][col]
                average_reward[step, agent_num] += reward
                
                # % optimal action
                if action == testbed.opt:
                    optimal_pulls[step, agent_num] += 1
                agent_num += 1
            
    # average values
    average_reward /= testbed.runs
    optimal_pulls /= testbed.runs
    
    graph_results(average_reward, optimal_pulls, agents)


def graph_results(average_reward, optimal_pulls, eps:list[eGreedy]) -> None:
    """Graph results based on Figure 2.2 of the textbook

    Args:
        average_reward (numpy.ndarray): xy plot for number of steps vs average reward
        optimal_pulls (numpy.ndarray): xy plot for number of steps vs % optimal action
        eps (list[eGreedy]): different epsilon values
    """

    fig, (ax1, ax2) = plt.subplots(2)
    
    # Graph 2.2: Average Reward
    ax1.plot(average_reward)
    ax1.set_ylabel("Average Reward")
    ax1.set_xlabel("Steps")
    ax1.legend(eps, loc="lower right")

    # Graph 2.2: % Optimal Action
    ax2.plot(optimal_pulls * 100)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% Optimal Action")
    ax2.set_xlabel("Steps")
    ax2.legend(eps, loc="lower right")
    
    plt.show()    
    # fig.savefig("comparison.png")
    
def main():
    testbed = TestBed(runs=1000)
    agents = []
    epsilon_vals = [0.1, 0.01, 0]
    for val in epsilon_vals:
        agents.append(eGreedy(testbed, val))
    # run the experiment!!
    run(agents, testbed, steps=500)

if __name__ == "__main__":
    main()