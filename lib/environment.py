"""
Defines an environment. The environment gives its current state, available actions based on the current state,
and reward based on action taken. It also changes the state based on the action taken.
"""

from errors import InvalidMoveError


class Board:
    def __init__(self, state):
        """
        Constructor for the Environment class.
        :param state: the initial state of the environment.
        """
        self.state = state
        self.action_space = {1: [2, 4], 2: [1, 3, 5], 3: [], 4: [1, 5], 5: [2, 4, 6], 6: [3, 5]}
        self.state_space = {1, 2, 3, 4, 5, 6}

    def move(self, dest):
        """
        Move the agent from current state to the destination state given by `dest` variable.
        :param dest: destination state.
        :return: reward obtained by performing the move.
        :raises InvalidMoveError: raises InvalidMoveError exception when the attempted move to destination is invalid.
        """
        if dest not in self.action_space[self.state]:
            raise InvalidMoveError()
        else:
            immediate_reward = self.get_immediate_reward(dest)
            self.state = dest
            return immediate_reward

    def get_immediate_reward(self, dest):
        """
        Gets immediate reward from moving from the current state to the `dest` state.
        :param dest: destination state.
        :return: immediate reward.
        """
        if self.state != 3 and dest == 3:
            return 100
        else:
            return 0
