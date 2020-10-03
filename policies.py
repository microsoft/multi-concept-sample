"""
Fixed policies to test our sim integration with. These are intended to take
Brain states and return Brain actions.
"""

import random

def random_policy(state):
    """
    Ignore the state, select randomly.
    """
    action = {
        'command': random.randint(1, 2)
    }
    return action

def coast(state):
    """
    Ignore the state, always select one exported brain.
    """
    action = {
        'command': 1
    }
    return action

POLICIES = {"random": random_policy,
            "coast": coast}