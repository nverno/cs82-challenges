#!/usr/bin/env python

"""
Run code to compute state values, and update the policy based on action values.
"""
from code import *

# Grid
dim_x = 10
dim_y = 10

# Actions
n_actions = 8
actions = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]
# action => index lookup table
action_index = {t: i for i,t in enumerate(actions)}

# Uniform initial policy
initial_policy = get_initial_policy(n_actions, dim_x, dim_y)

# Computing state values (long runtime)
sv_nruns = 10
G = MC_state_values(initial_policy, sv_nruns).reshape((10,10))

# Policy Improvement (long runtime)
pi_episodes = 10
pi_initial_state = 0
Q = np.zeros((100, 8))
Q = MC_action_values(initial_policy, Q, pi_episodes, pi_initial_state)
