"""
Module for training the agent in a naive way. This just calculates the state, state-action values, and optimum policy
using the Bellman equation.
"""

from collections import defaultdict

from .agent import Agent


class NaiveAgent(Agent):
    def __init__(self, gamma=0.9):
        """
        The agent is instantiated by using the discount factor.
        :param gamma: discount factor.
        """
        super().__init__(gamma)
        self.pi = {}
        self.q = defaultdict(dict)
        self.v = {}

    def calculate_state_value(self, board, state, visited=None):
        """
        Calculates the state values using the Bellman equation and stores them in the instance variable, v.
        :param board: Environment instance.
        :param state: state of which the value is to be calculated.
        :param visited: set of visited states.
        :return: state value of the state.
        """
        if not visited:
            board.state = state
            visited = {state}

        if not board.action_space[state]:
            return 0

        max_v_value = 0
        for dest in board.action_space[state]:

            if dest not in visited:
                immediate_reward = board.move(dest)
                visited.add(dest)
                v_value = immediate_reward + self.gamma * self.calculate_state_value(board, board.state, visited)
                if v_value >= max_v_value:
                    max_v_value = v_value
                board.state = state
                visited.remove(dest)

        return max_v_value

    def calculate_state_values(self, board):
        """
        Calculates state values of all the states in the state space.
        :param board: Environment instance.
        :return: the state values for all the states.
        """
        for state in board.state_space:
            self.v[state] = self.calculate_state_value(board, state)

        return self.v

    def calculate_state_action_values(self, board):
        """
        Calculate all the state-action values.
        :param board: Environment instance.
        :return: all the state action values.
        """
        for state in board.state_space:
            board.state = state

            best_policy = None
            best_value = -float('inf')

            for dest in board.action_space[state]:
                self.q[state][dest] = board.move(dest) + self.gamma * self.calculate_state_value(board, board.state)
                board.state = state
                if self.q[state][dest] > best_value:
                    best_policy = dest
                    best_value = self.q[state][dest]
            self.pi[state] = best_policy
        return self.q

    def learn(self, board):
        """
        Learn the state-action values.
        :param board: <Environment> Environment instance.
        :return: None
        """
        self.calculate_state_action_values(board)
