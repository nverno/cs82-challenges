#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import sign
import numpy.random as nr
import numpy.linalg as la
import matplotlib.pyplot as plt
from copy import deepcopy
import itertools

## -------------------------------------------------------------------
### Actions
raw_actions = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]

def vec_to_move(vec):
    v = np.array(vec, dtype=np.float)
    if (sum(abs(v)) > 1):
        v *= 0.707
    return v

def get_actions(actions=raw_actions):
    """Convert actions to numpy arrays, and create reverse lookup tables."""
    vecs_normalized = list(map(lambda x: vec_to_move(x), actions))
    return vecs_normalized, {a: i for i,a in enumerate(vecs_normalized)},\
        {i: a for i,a in enumerate(actions)}

def get_action_index(action):
    """Return index from action tuple."""
    return actions_index[action]

def get_index_action(index):
    """Return action tuple from index."""
    return index_actions[index]

actions, actions_index, index_actions = get_actions()

## -------------------------------------------------------------------
### Simulate

def is_terminal(loc):
    """Return True if x,y is in the terminal state."""
    x, y = loc
    return (y <= 0 and x >= 9) or (x >= 10 and y <= 1)


def sim_environment(loc, action):
    """
    Update current LOC with ACTION.
    :loc is a numpy array of (x, y) of current location
    :action is a numpy array representing movement vector (x, y)
    """
    new_loc = loc + action
    x, y = new_loc
    
    if is_terminal(new_loc):
        return new_loc, 1000, True
    
    # Returns to initial location if falls off of cliff
    if x >= 1 and x <= 9 and y >= 0 and y <= 1:
        return np.array([0,0]), -100, False

    # If hits a wall, stays in initial location
    if x < 0 or x > 10 or y < 0 or y > 10:
        return loc, -10, False

    return new_loc, -1, False
    
    
def test_sim(init, move):
    """
    Show simulator results for all movements from INIT state.
    :init numpy array (x, y)
    """
    fig = plt.figure(figsize=())
    ax = fig.gca()
    
    print(f"initial location: {init}")
    for m in moves:
        res, v, f = sim_environment(init, m)
        move = [round(i, 3) for i in m]
        res = [round(i, 3) for i in res]
        print(f"move: {move} ==> {res, v, f}")

        
# Some basic tests
# inits = [[0,0],                                   # start
#          [1,0],                                   # left of cliff
#          [9,1],                                   # right of cliff
#          [5,2],                                   # middle/above cliff
#          [5,5],                                   # middle of grid
#          [8,1],                                   # border terminal
#          [8,1]                                    # border terminal
#          ]
# inits = map(lambda x: np.array(x, dtype=np.float), inits)

# for i in inits:
#     test_sim(i, moves)

# def show_moves(start, actions=moves):
#     """Show outcome of moves from START location."""
#     fig, ax = draw_grid(10)
#     for a in actions:
#         l, v, t = sim_environment(start, a)

## -------------------------------------------------------------------
### Grid Approximation

def state(x, x_lims = (0.0,10.0), n_tiles = 10):
    """Function to compute tile state given positon"""
    state = int((x - x_lims[0])/(x_lims[1] - x_lims[0]) * float(n_tiles))
    if state > n_tiles - 1:
        state = n_tiles - 1
    return state

def grid_state(loc):
    """Convert x,y coordinates into single state value."""
    return state(loc[0]) + 10 * state(loc[1])

def grid_index(num):
    """Convert state to x,y coordinates in grid."""
    return num // 10, num % 10

# Monte Carlo State Value Estimation
def take_action(loc, policy, actions=actions):
    '''
    Function takes action given state using the transition probabilities 
    of the policy.
    '''
    ## Find the action given the transistion probabilities defined by the policy.
    x_grid, y_grid = state(loc[0]), state(loc[1])
    ind = nr.choice(8, p = policy[x_grid][y_grid])
    action = actions[ind]
    loc, reward, terminal = sim_environment(loc, action)
    return ind, action, loc, reward, terminal

## -------------------------------------------------------------------
### Initial Policy 

def get_initial_policy(nactions, x, y):
    """
    Generate initial policy given a number of actions and 
    x,y-dimensions of grid
    """
    # ensure lists aren't pointers to same memory
    return np.ones((x, y, nactions)) * 1./nactions

initial_policy = get_initial_policy(8, 10, 10)

## -------------------------------------------------------------------
### Single MC walks 

def MC_episode(policy, G, n_visits): 
    '''
    Function creates the Monte Carlo samples of one episode.
    This function does most of the real work
    '''
    ## For each episode we use a list to keep track of states we have visited.
    ## Once we visit a state we need to accumulate values to get the returns
    states_visited = np.zeros(100, dtype=np.bool)
    visited = []
    
    ## Find the starting state
    loc = np.array([0, 0])
    current_state = grid_state(loc)
    terminal = False

    while not terminal:
        ## Find the next action and reward
        ind, action, loc, reward, terminal = take_action(loc, policy)

        ## Add the reward to the states visited if this is a first visit  
        if (not states_visited[current_state]):
            ## Mark that the current state has been visited 
            states_visited[current_state] = True
            visited.append(current_state)
            ## Add the reward to states visited 
            for state in visited:
                n_visits[state] = n_visits[state] + 1.0
                G[state] = G[state] + (reward - G[state])/n_visits[state]
        
        ## Update the current state for next transition
        current_state = grid_state(loc)

    return G, n_visits


def MC_state_values(policy, n_episodes, verbose=False):
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
    
    ## Iterate over the episodes
    for i in range(n_episodes):
        if verbose: print(f"episode: {i}")
        G, n_visits = MC_episode(policy, G, n_visits)

    return G

## -------------------------------------------------------------------
### Examine state values

def show_state_values(G):
    """Show state values with origin in bottom left."""
    res = [x[::-1] for x in G]
    return res[::-1]


def plot_state_values(G):
    """Plot state values as image, with origin in bottom left."""
    res = [x[::-1] for x in G]
    plt.imshow(res, origin='lower')


def state_value_norm(G):
    """Euc. norm of state values."""
    return la.norm(G)

## -------------------------------------------------------------------
### Monte Carlo State Policy Improvment

def MC_action_value_episode(policy, Q, n_visits, inital_state, n_states, n_actions):
    '''
    Function creates the Monte Carlo samples of action values for one episode.
    This function does most of the real work.
    '''
    ## For each episode we use a list to keep track of states we have visited.
    ## Once we visit a state we need to accumulate values to get the returns
    state_actions_visited = np.zeros((n_states, n_actions), dtype=np.bool)
    current_state = initial_state
    loc_current = np.array(grid_index(current_state))
    terminal = False

    while not terminal:
        ## Find the next action and reward
        action_idx, action, loc_current, reward, terminal =\
            take_action(loc_current, policy)
        
        ## Check if this state-action has been visited.
        if not state_actions_visited[current_state, action_idx]:
            ## Mark that the current state-action has been visited 
            state_actions_visited[current_state, action_idx] = True
            ## This is first vist MS, so must loop over all state-action pairs and 
            ## add the reward and increment the count for the ones visited.
            for s,a in itertools.product(range(n_states), range(n_actions)):
                ## Add reward to if these has been a visit to the state
                if state_actions_visited[s, a] == True:
                    n_visits[s, a] += 1
                    Q[s, a] += (reward - Q[s, a])/n_visits[s, a]
        
        ## Update the current state for next transition
        current_state = grid_state(loc_current)
    
    return Q, n_visits

def MC_action_values(policy, Q, n_episodes, inital_state, verbose=False):
    '''
    Function evaluates the action-values given a policy for the specified number
    of episodes and initial state
    '''
    n_states = 100
    n_actions = 8
    ## Array to count visits to action-value pairs
    n_visits = np.zeros((n_states, n_actions))
    
    ## Take n_episodes MC samples
    for i in range(n_episodes):
        if verbose and i % 50 == 0: print(f"episode {i}")
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
# n_episodes = 5
# initial_state = 0
# Q = np.zeros((100, 8))
# Q = MC_action_values(initial_policy, Q, n_episodes, initial_state)

## -------------------------------------------------------------------
### Update Policy
def update_policy(policy, Q, epsilon, nstates=100, keys=raw_actions):
    '''
    Updates the policy based on estimates of Q using an epslion greedy
    algorithm. The action with the highest action value is used.
    '''
    ## Iterate over the states and find the maximm action value.
    for state in range(nstates):
        ## First find the index of the max Q values  
        q = Q[state,:]
        max_action_index = np.where(q == max(q))[0]

        ## Find the probabilities for the transitions
        n_transitions = float(len(q))
        n_max_transitions = float(len(max_action_index))
        p_max_transitions =\
            (1.0 - epsilon *(n_transitions - n_max_transitions))/(n_max_transitions)
        nx, ny = grid_index(state)
        
        ## Now assign the probabilities to the policy as epsilon greedy.
        for key in keys:
            index = get_action_index(key)
            if index in max_action_index:
                policy[nx][ny][index] = p_max_transitions
            else:
                policy[nx][ny][index] = epsilon

    return policy

## -------------------------------------------------------------------
### Iteratively updating policy
def MC_policy_improvement(policy, n_episodes, n_cycles, inital_state = 0, 
                          epsilon = 0.1, n_actions = 8):
    '''
    Function perfoms GPI using Monte Carlo value estimation.
    Updates to policy are epsilon greedy to prevent the algorithm
    from being trapped at some point.
    '''
    Q = np.zeros((len(policy), n_actions))

    for _ in range(n_cycles):
        Q = MC_action_values(policy, Q, n_episodes, inital_state)
        policy = update_policy(policy, Q, epsilon = epsilon)

    return policy

improved_policy = MC_policy_improvement(initial_policy, 1000, 5, epsilon = 0.1)  
for state in range(16):
    print(improved_policy[state])
    
## -------------------------------------------------------------------
### Plotting the policy

# def arrow_xy(action, prob):
    
def arrow_xy(i_action, prob):
    action = get_index_action(i_action)
    ## Position change given action:
    if(action == 0): # move right
        a_out = [1.0,0.0]
    elif(action == 1): # move right-up
        a_out = [0.707,0.707]
    elif(action == 2): # move up
        a_out = [0.0,1.0]
    elif(action == 3): # move left-up
        a_out = [-0.707,0.707]
    elif(action == 4): # move left
        a_out = [-1.0,0]
    elif(action == 5): # move left-down
        a_out = [-0.707,-0.707]
    elif(action == 6): # move down
        a_out = [0.0,-1.0]
    elif(action == 7): # move right-down
        a_out = [0.707,-0.707]
    return [prob*z for z in a_out]
