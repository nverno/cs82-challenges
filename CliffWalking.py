#!/usr/bin/env python
# coding: utf-8

# # Challenge Assignment
# ## Cliff Walking with Reinforcement Learning
# 
# ## CSCI E-82A
# 
# >**Make sure** you include your name along with the name of your team and team members in the notebook you submit.

# **Your name and team name here:** 

# ## Introduction
# 
# In this challenge you will apply Monte Carlo reinforcement learning algorithms to a classic problem in reinforcement learning, known as the **cliff walking problem**. The cliff walking problem is a type of game. The goal is for the agent to find the highest reward (lowest cost) path from a starting state to the goal.   
# 
# There are a number of versions of the cliff walking problems which have been used as research benchmarks over the years. You can find a short discussion of the cliff walking problem on page 132 of Sutton and Barto, second edition.    
# 
# In the general cliff walking problem the agent starts in one corner of the state-space and must travel to goal, or terminal state, in another corner of the state-space. Between the starting state and goal state there is an area with a **cliff**. If the agent falls off a cliff it is sent back to the starting state. A schematic diagram of the state-space is shown in the diagram below.      
# 
# <img src="CliffWalkingDiagram.JPG" alt="Drawing" style="width:500px; height:400px"/>
# <center> State-space of cliff-walking problem </center>
# 
# 

# ### Problem Description
# 
# The agent must learn a policy to navigate from the starting state to the terminal state. The properties this problem are as follows:
# 
# 1. The state-space has two **continuous variables**, x and y.
# 2. The starting state is at $x = 0.0$, $y = 0.0$. 
# 3. The terminal state has two segments:
#   - At $y = 0.0$ is in the range $9.0 \le x \le 10.0$. 
#   - At $x = 10.0$ is in the range $0.0 \le y \le 1.0$.  
# 4. The cliff zone is bounded by:
#   - $0.0 \le y \le 1.0$ and 
#   - $1.0 \le x \le 9.0$. 
# 5. An agent entering the cliff zone is returned to the starting state.
# 6. The agent moves 1.0 units per time step. 
# 7. The 8 possible **discrete actions** are moves in the following directions:  
#   - +x, 
#   - +x, +y,
#   - +y
#   - -x, +y,
#   - -y,
#   - -x, -y,
#   - -y, and
#   - +x, -y. 
# 8. The rewards are:
#   - -1 for a time step in the state-space,
#   - -10 for colliding with an edge (barrier) of the state-space,
#   - -100 for falling off the cliff and returning to the starting state, and 
#   - +1000 for reaching the terminal or goal state. 
#   
# 

# ## Instructions
# 
# In this challenge you and your team will do the following. Include commentary on each component of your algorithms. Make sure you answer the questions.  

# ### Environment Simulator   
# 
# Your reinforcement learning agent cannot contain any information about the environment other that the starting state and the possible actions. Therefore, you must create an environment simulator, with the following input and output:
# - Input: Arguments of state, the $(x,y)$ tuple, and discrete action
# - Output: the new state (s'), reward, and if the new state meets the terminal or goal criteria.
# 
# Make sure you test your simulator functions carefully. The test cases must include, steps with each of the actions, falling off the cliff from each edge, hitting the barriers, and reaching the goal (terminal) edges. Errors in the simulator will make the rest of this challenge difficult.   
# 
# > **Note**: For this problem, coordinate state is represented by a tuple of continuous variables. Make sure that you maintain coordinate state as continuous variables for this problem. 

# In[64]:

import numpy as np
import pandas as pd
import numpy.random as nr
import matplotlib.pyplot as plt
from copy import deepcopy
import itertools

# get_ipython().run_line_magic('matplotlib', 'inline')

def sim_environment(x, y, action):
    fell = False
    hit_wall = False
    # calculate new location
    x_prime = x
    y_prime = y
    if (action[0] == 1):
        if (action[1] == 1):
            x_prime = x + .707
            y_prime = y + .707
        elif(action[1] == -1):
            x_prime = x + .707
            y_prime = y - .707
        else:
            x_prime = x + 1
    elif (action[0] == -1):
        if (action[1] == 1):
            x_prime = x - .707
            y_prime = y + .707
        elif(action[1] == -1):
            x_prime = x - .707
            y_prime = y - .707
        else:
            x_prime = x - 1
    else:
        if (action[1] == 1):
            y_prime = y + 1
        elif (action[1] == -1):
            y_prime = y - 1
             
    #ensure new location is in bounds
    if (x_prime > 10.0):
        x_prime = 0.0
        y_prime = 0.0
        hit_wall = True
    elif (x_prime < 0.0):
        x_prime = 0.0
        y_prime = 0.0
        hit_wall = True
    if (y_prime > 10.0):
        x_prime = 0.0
        y_prime = 0.0
        hit_wall = True
    elif (y_prime < 0.0):
        x_prime = 0.0
        y_prime = 0.0  
        hit_wall = True
             
    #ensure new location is not off cliff
    if (x_prime >= 1.0 and x_prime <= 9.0):
        if (y_prime >= 0.0 and y_prime <= 1.0):
             x_prime = 0.0
             y_prime = 0.0
             fell = True
    
    ## At the terminal state or not and set reward
    if (in_terminal(x_prime, y_prime)):
        done = True
        reward = 1000
    elif (fell):
        done = False
        reward = -100
    elif (hit_wall):
        done = False
        reward = -10
        x_prime = x
        y_prime = y
    else:
        done = False
        reward = -1.0
    # output new state (s'), reward, and if the new state meets the terminal or goal criteria
    return(x_prime, y_prime, done, reward)

def in_terminal(x, y):
    if (y == 0.0 and x >= 9.0):
        return True
    elif (x == 10.0 and y <= 1.0):
        return True
    return False


# ### Grid Approximation
# 
# The state-space of the cliff walking problem is continuous. Therefor, you will need to use a **grid approximation** to construct a policy. The policy is specified as the probability of action for each grid cell. For this problem, use a 10x10 grid. 
# 
# > **Note:** While the policy uses a grid approximation, state should be represented as continuous variables.

# In[75]:


def state(x, x_lims = (0.0,10.0), n_tiles = 10):
    """Function to compute tile state given positon"""
    state = int((x - x_lims[0])/(x_lims[1] - x_lims[0]) * float(n_tiles))
    if(state > n_tiles - 1): state = n_tiles - 1
    return(state)

def get_grid_state(x, y):
    return (state(x) + 10 * state(y))

def get_index(num):
    return num % 10, num // 10


# ### Initial Policy
# 
# Start with a uniform initial policy. A uniform policy has an equal probability of taking any of the 8 possible actions for each cell in the grid representation.     
# 
# > **Note:** As has already been stated, the coordinate state representation for this problem is a tuple of coordinate values. However, policy, state-values and action-values are represented with a grid approximation. 
# 
# > **Hint:** You may wish to use a 3-dimensional numpy array to code the policy for this problem. With 8 possible actions, this approach will be easier to work with. 
# 
# 

# In[71]:


initial_policy = [[[.125 for i in range(8)] for j in range(10)] for k in range(10)]


# ### Monte Carlo State Value Estimation   
# 
# For the initial uniform policy, compute the state values using the Monte Carlo RL algorithm:
# 1. Compute and print the state values for each grid in the representation. Use at least 1,000 episodes. This will take some time to execute.      
# 2. Plot the grid of state values, as an image (e.g. matplotlib [imshow](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html)). 
# 3. Compute the Forbenious norm (Euclidean norm) of the state value array with [numpy.linalg.norm](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html). You will use this figure as a basis to compare your improved policy. 
# 
# Study your plot to ensure your state values seem correct. Do these state values seem reasonable given the uniform policy and why? Make sure you pay attention to the state values of the cliff zone.    
# 
# > **Hint:** Careful testing at each stage of your algorithm development will potentially save you considerable time. Test your function(s) to for a single episode to make sure your algorithm converges. Then test for say 10 episodes to ensure the state values update in a reasonable manner at each episode.    
# 
# > **Note:** The Monte Carlo episodes can be executed in parallel for production systems. The Markov chain of each episode is statistically independent. 

# In[66]:


#import matplotlib.pyplot as plt
actions = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]
    
def take_action(x, y, policy):
    '''Function takes action given state using the transition probabilities 
    of the policy'''
    ## Find the action given the transistion probabilities defined by the policy.
    x_grid = state(x)
    y_grid = state(y)
    action = actions[nr.choice(8, p = policy[x_grid][y_grid])]
    x_prime, y_prime, is_terminal, reward = sim_environment(x,y, action)
    return (action, x_prime, y_prime, is_terminal, reward)

#print(initial_policy[0][0])
#take_action(0,0,initial_policy)

def MC_episode(policy, G, n_visits): 
    '''Function creates the Monte Carlo samples of one episode.
    This function does most of the real work'''
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
    while(not terminal):
        ## Find the next action and reward
        action, x_prime, y_prime, terminal, reward = take_action(x_current, y_current, policy)
        #print("from: " + str((x_current, y_current, action)))
        #print("to: " + str((x_prime, y_prime, is_terminal, reward)))
        ## Add the reward to the states visited if this is a first visit  
        if(states_visited[current_state] == 0):
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
    '''Function that evaluates the state value of 
    a policy using the Monte Carlo method.'''
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
        print ("running episode: " + str(i))
    return(G) 

MC_state_values(initial_policy, 100).reshape((10,10))
#actions[nr.choice(8, p = initial_policy[0][0]) + 1]


# In[ ]:





# ANS:

# ### Monte Carlo State Policy Improvement   
# 
# Finally, you will perform Monte Carlo RL policy improvement:
# 1. Starting with the uniform policy, compute action-values for each grid in the representation. Use at least 1,000 episodes.      
# 2. Use these action values to find an improved policy.
# 3. To evaluate your updated policy compute the state-values for this policy.  
# 4. Plot the grid of state values for the improved policy, as an image. 
# 5. Compute the Forbenious norm (Euclidean norm) of the state value array. 
# 
# Compare the state value plot for the improved policy to the one for the initial uniform policy. Does the improved state values increase generally as distance to the terminal states decreases?  Is this what you expect and why?    
# 
# Compare the norm of the state values with your improved policy to the norm for the uniform policy. Is the increase significant?  
# 
# > **Hint:** Careful testing at each stage of your algorithm development will potentially save you considerable time. Test your function(s) to for a single episode to make sure your algorithm converges. Then test for say 10 episodes to ensure the state values update in a reasonable manner at each episode.   
# 
# > **Note:** You could continue to improve policy using the general policy improvement algorithm (GPI). In the interest of time, you are not required to do so here. 

# In[67]:


def get_action_index(action):
    if (action == (0,1)): return 0
    elif (action == (0,-1)): return 1
    elif (action == (1,0)): return 2
    elif (action == (1,1)): return 3
    elif (action == (1,-1)): return 4
    elif (action == (-1,0)): return 5
    elif (action == (-1,1)): return 6
    elif (action == (-1,-1)): return 7
    
def print_Q(Q):
    Q = pd.DataFrame(Q, columns = ['N', 'S', 'E', 'NE', 'SE', 'W', 'NW', 'SW'])
    print(Q)
    
def MC_action_value_episode(policy, Q, n_visits, inital_state, n_states, n_actions, action_index = {'u':0, 'd':1, 'l':2, 'r':3}):
    '''Function creates the Monte Carlo samples of action values for one episode.
    This function does most of the real work'''
    ## For each episode we use a list to keep track of states we have visited.
    ## Once we visit a state we need to accumulate values to get the returns
    state_actions_visited = np.zeros((n_states, n_actions))
    x_current = 0.0
    y_current = 0.0
    current_state = get_grid_state(x_current, y_current)
    terminal = False  
    while(not terminal):
        ## Find the next action and reward
        #action, s_prime, reward, terminal = take_action(current_state, policy)
        action, x_prime, y_prime, terminal, reward = take_action(x_current, y_current, policy)
        action_idx = get_action_index(action)         
        
        ## Check if this state-action has been visited.
        if(state_actions_visited[current_state, action_idx] != 1.0):
            ## Mark that the current state-action has been visited 
            state_actions_visited[current_state, action_idx] = 1.0  
            ## This is first vist MS, so must loop over all state-action pairs and 
            ## add the reward and increment the count for the ones visited.
            for s,a in list(itertools.product(range(n_states), range(n_actions))):
                ## Add reward to if these has been a visit to the state
                if(state_actions_visited[s,a] == 1.0):
                    n_visits[s,a] = n_visits[s,a] + 1.0
                    Q[s,a] = Q[s,a] + (reward - Q[s,a])/n_visits[s,a]    
        ## Update the current state for next transition
        current_state = get_grid_state(x_prime, y_prime)
        x_current = x_prime
        y_current = y_prime
    return (Q, n_visits) 

def MC_action_values(policy, Q, n_episodes, inital_state):
    '''Function evaluates the action-values given a policy for the specified number of episodes and 
    initial state'''
    n_states = 100
    n_actions = 8
    ## Array to count visits to action-value pairs
    n_visits = np.zeros((n_states, n_actions))
    ## Dictionary to hold neighbor states
    neighbors = {}
    
    ## Loop over number of episodes
    for i in range(n_episodes):
        ## One episode of MC
        print ("running episode #: " + str(i))
        Q, n_visits = MC_action_value_episode(policy, Q, n_visits, initial_state, n_states, n_actions)
    return(Q)

n_episodes = 5
initial_state = 0
Q = np.zeros((100, 8))
Q = MC_action_values(initial_policy, Q, n_episodes, initial_state)
print_Q(Q)


# In[73]:

def update_policy(policy, Q, epsilon):
    '''Updates the policy based on estiamtes of Q using 
    an epslion greedy algorithm. The action with the highest
    action value is used.'''
    
    ## Find the keys for the actions in the policy
    keys = actions
    
    ## Iterate over the states and find the maximm action value.
    for state in range(len(policy)):
        ## First find the index of the max Q values  
        q = Q[state,:]
        max_action_index = np.where(q == max(q))[0]
        
        ## Find the probabilities for the transitions
        n_transitions = float(len(q))
        n_max_transitions = float(len(max_action_index))
        p_max_transitions = (1.0 - epsilon *(n_transitions - n_max_transitions))/(n_max_transitions)
        x_idx = state % 10
        y_idx = state // 10
        
        ## Now assign the probabilities to the policy as epsilon greedy.
        for key in keys:
            index = get_action_index(key)
            if(index in max_action_index): policy[x_idx][y_idx][index] = p_max_transitions
            else: policy[x_idx][y_idx][index] = epsilon
    return(policy)


# In[74]:


update_policy(deepcopy(initial_policy), Q, 0.1)


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from itertools import product
get_ipython().run_line_magic('matplotlib', 'inline')

def get_action_index(action):
    if (action == (0,1)): return 0
    elif (action == (0,-1)): return 1
    elif (action == (1,0)): return 2
    elif (action == (1,1)): return 3
    elif (action == (1,-1)): return 4
    elif (action == (-1,0)): return 5
    elif (action == (-1,1)): return 6
    elif (action == (-1,-1)): return 7
    
    
def arrow_xy(action, prob):
    ## Position change given action:
    if(action == 0): # move right
        a_out = [0, 1]
    elif(action == 1): # move right-up
        a_out = [0, -1]
    elif(action == 2): # move up
        a_out = [1, 0]
    elif(action == 3): # move left-up
        a_out = [1, 1]
    elif(action == 4): # move left
        a_out = [1, -1]
    elif(action == 5): # move left-down
        a_out = [-1, 0]
    elif(action == 6): # move down
        a_out = [-1, 1]
    elif(action == 7): # move right-down
        a_out = [-1, -1]
    return [prob*z for z in a_out]

def display_policy(policy, num_cells = 10, lims = 10.0):
    '''
    Funciton plots arrows in the direction of actions with length 
    of arrows proportional to probability

    Policy argument must be a numpy array with dimensions:
    grid x-dimension cell index
    grid y-dimension cell index
    acton index
    '''
    ## Find max arrow length 
    cell_width = lims/float(num_cells)
    arrow_length = 0.5 * cell_width

    ## Set up the figure area
    fig = plt.figure(figsize=(8, 8)) # define plot area
    ax = fig.gca() # define axis 

    ## Iterate over the grid and plot
    for x_indx, y_indx, action in product(range(num_cells),range(num_cells),range(8)):
        ## Find endpoints of the line segment
        move = arrow_xy(action, policy[x_indx][y_indx][action])
        x0 = x_indx * cell_width + arrow_length
        x1 = x0 + move[0]
        y0 = y_indx * cell_width + arrow_length
        y1 = y0 + move[1]
        ## now plot 
        # ax.plot([x0,x1],[y0,y1])
        ax.annotate(' ', xytext =(x0,y0), xy = (x1,y1), arrowprops = ({'arrowstyle':'->'}))

    ## Now plot the grid
    ## first the vertical lines
    for indx in range(num_cells-1):
        x = cell_width *(indx+1)
        ax.plot([x,x],[0.0,lims], linestyle='dashed', color='green')
    ## now the horrizontal lines
    for indx in range(num_cells-1):
        y = cell_width *(indx+1)
        ax.plot([0.0,lims],[y,y], linestyle='dashed', color='green') 
    ## and boundaries
    ax.plot([0.0,0.0], [0.0,lims], color = 'black')
    ax.plot([0.0,lims], [0.0,0.0], color = 'black')
    ax.plot([lims,0.0], [lims,lims], color = 'black')
    ax.plot([lims,lims], [0.0,lims], color = 'black')


# In[50]:


display_policy(np.array(initial_policy))


# ANS:
# 
# ANS:
# 
# ANS:

# ## Solution
# 
# Create cells below for your solution to the stated problem. Be sure to include some Markdown text and code comments to explain each component of your algorithm. 

# In[ ]:




