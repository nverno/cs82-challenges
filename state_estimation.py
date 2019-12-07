#!/usr/bin/env python
import numpy.random as nr
import numpy as np
from numpy import sign
from numpy import sign
from itertools import permutations as perm

actions = list(perm([0, 1, -1], 2)) + [[1, 1], [-1, -1]]
# actions = [(0,0), (0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]
initial_actions = [.125]*8
x_actions = [initial_actions] * 10
initial_policy = [x_actions] * 10

# Simulation
def is_terminal(x, y):
    if (y <= 0 and (x >= 9.0 or x <= 10)):
        return True
    elif (x >= 10 and (y >=0 or y <= 1.0)):
        return True
    return False

def sim_environment(x, y, action):
    px = x
    py = y
    
    if (sum([abs(i) for i in action]) > 1):
        action = [sign(a) * .707 for a in action]

    x += action[0]
    y += action[1]

    if (is_terminal(x, y)):
        return x, y, 1000, True
    
    # Check boundary conditions
    if (x < 0 or y > 9 or y < 0 or x > 9):
        return px, py, -10, False

    # check cliff
    if (x >= 1 and y <= 9 and x >= 0 and y <= 1):
        return 0, 0, -100, False

    return x, y, -1, False

# Grid Approximation
def state(x, x_lims = (0.0,10.0), n_tiles = 10):
    """Function to compute tile state given positon"""
    state = int((x - x_lims[0])/(x_lims[1] - x_lims[0]) * float(n_tiles))
    if (state > n_tiles - 1): state = n_tiles - 1
    return state

def get_grid_state(x, y):
    return state(x) + 10 * state(y)

# Value Estimation
def take_action(x, y, policy):
    '''Function takes action given state using the transition probabilities 
    of the policy'''
    ## Find the action given the transistion probabilities defined by the policy.
    x_grid = state(x)
    y_grid = state(y)
    action = actions[nr.choice(8, p = policy[x_grid][y_grid]) + 1]
    x_prime, y_prime, is_terminal, reward = sim_environment(x, y, action)
    return (action, x_prime, y_prime, is_terminal, reward)

#print(initial_policy[0][0])
#take_action(0,0,initial_policy)

def MC_episode(policy, G, n_visits): 
    '''Function creates the Monte Carlo samples of one episode.
    This function does most of the real work'''
    ## For each episode we use a list to keep track of states we have visited.
    ## Once we visit a state we need to accumulate values to get the returns
    states_visited = []
        
    ## Find the starting state
    x_current = 0.0
    y_current = 0.0
    current_state = get_grid_state(x_current, y_current)
    terminal = False
    g = 0.0
    
    while (not terminal):
        ## Find the next action and reward
        action, x_prime, y_prime, is_terminal, reward =\
            take_action(x_current, y_current, policy)
        print("from: " + str((action, x_current, y_current, is_terminal, reward)))
        print("to: " + str((action, x_prime, y_prime, is_terminal, reward)))
        ## Add the reward to the states visited if this is a first visit  
        if (current_state not in states_visited):
            ## Mark that the current state has been visited 
            states_visited.append(current_state) 
            ## Add the reward to states visited 
            for state in states_visited:
                n_visits[state] = n_visits[state] + 1.0
                G[state] = G[state] + (reward - G[state])/n_visits[state]
        
        ## Update the current state for next transition
        current_state = get_grid_state(x_prime, y_prime) 
        x_current = x_prime
        y_current = y_prime
    return (G, n_visits) 

def MC_state_values(policy, n_episodes):
    '''Function that evaluates the state value of 
    a policy using the Monte Carlo method.'''
    ## Create list of states 
    n_states = 100
    
    ## An array to hold the accumulated returns as we visit states
    G = np.zeros((n_states))
    
    ## An array to keep track of how many times we visit each state so we can 
    ## compute the mean
    n_visits = np.zeros((n_states))
    print(n_visits)
    
    ## Iterate over the episodes
    for i in range(n_episodes):
        G, n_visits = MC_episode(policy, G, n_visits) # neighbors, i, n_states)
    return(G) 

MC_state_values(initial_policy, 1)
#actions[nr.choice(8, p = initial_policy[0][0]) + 1]
