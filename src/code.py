#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import sign
import numpy.random as nr
import numpy.linalg as la
import matplotlib.pyplot as plt
from copy import deepcopy

## -------------------------------------------------------------------
### Environment Simulator
 
def vec_to_move(vec):
    v = np.array(vec, dtype=np.float)
    if (sum(abs(v)) > 1):
        v *= 0.707
    return v

actions =\
    list(map(lambda x: vec_to_move(x),
             [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]))
    
def is_terminal(x, y):
    if (y <= 0 and x >= 9.0):
        return True
    elif (x >= 10 and y <= 1.0):
        return True
    return False

def sim_environment(loc, action):
    """
    Update current LOC with ACTION.
    :loc is a numpy array of (x, y)
    :action is a numpy array representing movement vector (x, y)
    """
    prev = loc
    loc = loc + action

    # Check boundary conditions
    if (loc[0] < 0 or loc[0] > 9 or loc[1] < 0 or loc[1] > 9):
        return prev, -10, False

    # check cliff
    if (loc[0] >= 1 and loc[0] <= 9 and loc[0] <= 1):
        return np.array([0,0]), -100, False

    if (is_terminal(loc[0], loc[1])):
        return loc, 1000, True
    
    return loc, -1, False

## -------------------------------------------------------------------
### Grid Approximation

def state(x, x_lims = (0.0,10.0), n_tiles = 10):
    """Function to compute tile state given positon"""
    state = int((x - x_lims[0])/(x_lims[1] - x_lims[0]) * float(n_tiles))
    if(state > n_tiles - 1): state = n_tiles - 1
    return(state)

def get_grid_state(x, y):
    """Convert x,y coordinates into single state value."""
    return (state(x) + 10 * state(y))

def get_index(num):
    """Convert state to x,y coordinates in grid."""
    return num % 10, num // 10

## -------------------------------------------------------------------
### Initial Policy 

initial_policy = [[[.125 for i in range(8)]
                   for j in range(10)]
                  for k in range(10)]
# initial_policy[0]

## -------------------------------------------------------------------
### Manange actions

actions = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]
action_index = {t: i for i,t in enumerate(actions)}

def get_action_index(action):
    return action_index[(action)]

# Monte Carlo State Value Estimation
def take_action(x, y, policy):
    '''Function takes action given state using the transition probabilities 
    of the policy'''
    ## Find the action given the transistion probabilities defined by the policy.
    x_grid = state(x)
    y_grid = state(y)
    action = actions[nr.choice(8, p = policy[x_grid][y_grid])]
    loc, is_terminal, reward =\
        sim_environment(np.array([x_grid, y_grid]), action)
    return (action, loc[0], loc[1], is_terminal, reward)

## -------------------------------------------------------------------
### Single MC walks 

def MC_episode(policy, G, n_visits): 
    '''
    Function creates the Monte Carlo samples of one episode.
    This function does most of the real work
    '''
    ## For each episode we use a list to keep track of states we have visited.
    ## Once we visit a state we need to accumulate values to get the returns
    states_visited = np.zeros(100)
    visited = []
    
    ## Find the starting state
    x_current = 0.0
    y_current = 0.0
    current_state = get_grid_state(x_current, y_current)
    terminal = False
    g = 0.0
    #i = 0
    #plt.axis([0, 10, 0, 10])
    #plt.scatter(x_current, y_current)
    while (not terminal):
        ## Find the next action and reward
        action, x_prime, y_prime, terminal, reward = \
            take_action(x_current, y_current, policy)
        #print("from: " + str((x_current, y_current, action)))
        #print("to: " + str((x_prime, y_prime, is_terminal, reward)))
        ## Add the reward to the states visited if this is a first visit  
        if (states_visited[current_state] == 0):
            ## Mark that the current state has been visited 
            states_visited[current_state] = 1
            visited.append(current_state)
            ## Add the reward to states visited 
            for state in visited:
                n_visits[state] = n_visits[state] + 1.0
                G[state] = G[state] + (reward - G[state])/n_visits[state]
        
        ## Update the current state for next transition
        current_state = get_grid_state(x_prime, y_prime) 
        x_current = x_prime
        y_current = y_prime
        #i = i + 1
        #plt.scatter(x_current, y_current)
    #plt.show()
    return (G, n_visits) 

def MC_state_values(policy, n_episodes):
    """
    Function that evaluates the state value of 
    a policy using the Monte Carlo method.
    """
    ## Create list of states 
    n_states = 100
    
    ## An array to hold the accumulated returns as we visit states
    G = np.zeros((n_states))
    
    ## An array to keep track of how many times we visit each state so we can 
    ## compute the mean
    n_visits = np.zeros((n_states))
    #print(n_visits)
    
    ## Iterate over the episodes
    for i in range(n_episodes):
        G, n_visits = MC_episode(policy, G, n_visits) # neighbors, i, n_states)
        # print ("running episode: " + str(i))
    return G

## -------------------------------------------------------------------
### Examine state values

# note: takes long time
n_runs = 10
G = MC_state_values(initial_policy, n_runs).reshape((10,10))

# Show state values with origin in bottom left
def show_state_values(G):
    """Show state values with origin in bottom left."""
    res = [x:[::-1] for x in G]
    return res[::-1]

# plot as image
def plot_state_values(G):
    """Plot state values as image, with origin in bottom left."""
    res = [x[::-1] for x in G]
    plt.imshow(res, origin='lower')

# Compute norm
dist = la.norm(G)

## -------------------------------------------------------------------
### Monte Carlo State Policy Improvment

def MC_action_value_episode(policy, Q, n_visits, inital_state, n_states, n_actions):
    '''
    Function creates the Monte Carlo samples of action values for one episode.
    This function does most of the real work.
    '''
    ## For each episode we use a list to keep track of states we have visited.
    ## Once we visit a state we need to accumulate values to get the returns
    state_actions_visited = np.zeros((n_states, n_actions))
    x_current = 0.0
    y_current = 0.0
    current_state = get_grid_state(x_current, y_current)
    terminal = False  

    while not terminal:
        ## Find the next action and reward
        #action, s_prime, reward, terminal = take_action(current_state, policy)
        action, x_prime, y_prime, terminal, reward =\
            take_action(x_current, y_current, policy)
        action_idx = get_action_index(action)         
        
        ## Check if this state-action has been visited.
        if state_actions_visited[current_state, action_idx] != 1.0:
            ## Mark that the current state-action has been visited 
            state_actions_visited[current_state, action_idx] = 1.0  
            ## This is first vist MS, so must loop over all state-action pairs and 
            ## add the reward and increment the count for the ones visited.
            for s,a in itertools.product(range(n_states), range(n_actions)):
                ## Add reward to if these has been a visit to the state
                if state_actions_visited[s,a] == 1.0:
                    n_visits[s,a] = n_visits[s,a] + 1.0
                    Q[s,a] = Q[s,a] + (reward - Q[s,a])/n_visits[s,a]
        
        ## Update the current state for next transition
        current_state = get_grid_state(x_prime, y_prime)
        x_current = x_prime
        y_current = y_prime
    
    return (Q, n_visits)

def MC_action_values(policy, Q, n_episodes, inital_state, verbose=False):
    '''
    Function evaluates the action-values given a policy for the specified number
    of episodes and initial state
    '''
    n_states = 100
    n_actions = 8
    ## Array to count visits to action-value pairs
    n_visits = np.zeros((n_states, n_actions))
    ## Dictionary to hold neighbor states
    neighbors = {}
    
    ## Loop over number of episodes
    for i in range(n_episodes):
        ## One episode of MC
        if verbose and i % 10 == 0: print (f"running episode #{i}")
        Q, n_visits =\
            MC_action_value_episode(
                policy, Q, n_visits, initial_state, n_states, n_actions)
    return Q

def print_Q(Q):
    """Pretty print state values."""
    Q = pd.DataFrame(Q, columns = ['N', 'S', 'E', 'NE', 'SE', 'W', 'NW', 'SW'])
    print(Q)
    
## -------------------------------------------------------------------
### Compute action-values, find Q 
# takes long time
n_episodes = 5
initial_state = 0
Q = np.zeros((100, 8))
Q = MC_action_values(initial_policy, Q, n_episodes, initial_state)
