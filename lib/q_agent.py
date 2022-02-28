"""
Module for training the agent using Q learning algorithm.
"""

from collections import defaultdict

import numpy as np

from .agent import Agent


class QLearningAgent(Agent):
    def __init__(self, gamma=0.9, epsilon=0.1):
        """
        The agent is instantiated by using the discount factor and epsilon.
        :param gamma: <float> discount factor.
        :param epsilon: <float> percentage of times when the agent wants to explore.
        """
        super().__init__(gamma, epsilon)
        self.pi = {}
        self.q = defaultdict(dict)
        self.v = {}

    def initialize_weights(self, board):
        """
        Initialize the state-action values to random values between 0 and 1.
        :param board: <Environment> Environment instance.
        :return: None
        """
        for state in board.state_space:
            for dest in board.action_space[state]:
                self.q[state][dest] = np.random.rand()

    def get_action(self, board, state):
        """
        Follows e-greedy policy to select the action give the board and state.
        :param board: <Environment> Environment instance.
        :param state: <int> current state of the environment.
        :return: <int> action given the state and board.
        """
        prob = np.random.rand()
        if prob >= self.epsilon:
            max_state_action_value = max(self.q[state].values())
            choices = [
                action for action in board.action_space[state] if self.q[state][action] == max_state_action_value
            ]
        else:
            choices = board.action_space[state]
        return np.random.choice(choices)

    def _learn(self, board):
        """
        Method to do one iteration of q-learning algorithm.
        :param board: <Environment> Environment instance.
        :return: None
        """
        state = 1
        board.state = 1
        while True:
            dest = self.get_action(board, state)
            immediate_reward = board.move(dest)

            self.q[state][dest] = immediate_reward + self.gamma * max(self.q[dest][next_dest] for next_dest in board.action_space[dest])
            state = dest
            if board.game_over:
                break

    def learn(self, board, num_games=100):
        """
        Learn the state-action values by using Q learning reinforcement learning algorithm.
        :param num_games: <int> number of games for agent to play.
        :param board: <Environment> Environment instance.
        :return: None
        """
        self.initialize_weights(board)

        for _ in range(num_games):
            self._learn(board)

        for state in board.state_space:
            self.pi[state] = max((self.q[state][dest], dest) for dest in self.q[state])[1]
