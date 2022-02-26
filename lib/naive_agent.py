from collections import defaultdict
from pprint import pprint

from environment import Board
from display import display_state_values, display_state_action_values


class Agent:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.q = defaultdict(dict)
        self.v = {}

    def calculate_state_value(self, board, state, visited=None):
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
        for state in board.state_space:
            self.v[state] = self.calculate_state_value(board, state)
        pprint("State values:")
        pprint(self.v)
        display_state_values(self)
        return self.v

    def calculate_state_action_values(self, board):
        for state in board.state_space:
            board.state = state
            for dest in board.action_space[state]:
                self.q[state][dest] = board.move(dest) + self.gamma * self.calculate_state_value(board, board.state)
                board.state = state
        display_state_action_values(self)
        return self.q


board = Board(4)
agent = Agent()
agent.calculate_state_action_values(board)



