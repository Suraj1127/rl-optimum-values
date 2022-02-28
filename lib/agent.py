"""
This module contains the generic agent.
"""


class Agent:

    def __init__(self, gamma=0.9, epsilon=0.1):
        """
        The agent is instantiated by using the discount factor and epsilon.
        :param gamma: <float> discount factor.
        :param epsilon: <float> percentage of times when the agent wants to explore.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.pi = None

    def learn(self, board):
        pass

    def determine_optimum_policy(self, board):
        """
        Finds the optimum policy by using the state-action values and returns it.
        :param board: Environment instance of the board.
        :return: optimum policy using the state-action values for all the states.
        """
        if self.pi:
            return self.pi
        else:
            self.learn(board)
            return self.determine_optimum_policy(board)
