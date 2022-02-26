"""
Module for display purpose of state values, state action values, and optimum policy.
Thinking of drawing boxes for the display purpose in the future.
"""
from pprint import pprint


def display_state_values(agent):
    print("State values:")
    pprint(agent.v)


def display_state_action_values(agent):
    print("State action values:")
    pprint(agent.q)


def display_optimum_policy(agent):
    print("Optimum policy:")
    pprint(agent.pi)
