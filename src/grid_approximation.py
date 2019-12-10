#!/usr/bin/env python

## -------------------------------------------------------------------
### Grid Approx.

def state(x, x_lims = (0.0,10.0), n_tiles = 10):
    """Function to compute tile state given positon"""
    state = int((x - x_lims[0])/(x_lims[1] - x_lims[0]) * float(n_tiles))
    if(state > n_tiles - 1): state = n_tiles - 1
    return(state)

def get_grid_state(x, y):
    """Convert x,y coordinates into single state value."""
    return (state(x) + 10 * state(y))

def grid_index(num):
    """Convert state to x,y coordinates in grid."""
    return num % 10, num // 10

def take_action(x, y, policy):
    '''Function takes action given state using the transition probabilities 
    of the policy'''
    ## Find the action given the transistion probabilities defined by the policy.
    x_grid = state(x)
    y_grid = state(y)
    action = actions[nr.choice(8, p = policy[x_grid][y_grid])]
    x_prime, y_prime, is_terminal, reward = sim_environment(x,y, action)
    return (action, x_prime, y_prime, is_terminal, reward)
