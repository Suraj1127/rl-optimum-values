from lib.display import display_state_action_values, display_optimum_policy
from lib.environment import Board
from lib.naive_agent import NaiveAgent
from lib.q_agent import QLearningAgent


if __name__ == "__main__":
    print("Naive learning agent:")
    board = Board()
    naive_agent = NaiveAgent()
    naive_agent.learn(board)
    display_state_action_values(naive_agent)
    display_optimum_policy(naive_agent)

    print()

    print("Q learning agent:")
    board = Board()
    q_agent = QLearningAgent()
    q_agent.learn(board, num_games=100000)
    display_state_action_values(q_agent)
    display_optimum_policy(q_agent)
