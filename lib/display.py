from pprint import pprint


def display_state_values(agent):
    print("State values:")
    pprint(agent.v)


def display_state_action_values(agent):
    print("State action values:")
    pprint(agent.q)