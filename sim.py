#!/usr/bin/env python
# coding: utf-8
from numpy import sign

def is_terminal(x, y):
    if (y <= 0 and (x >= 9.0 or x <= 10)):
        return True
    elif (x >= 10 and (y >=0 or y <= 1.0)):
        return True
    return False

def sim_environment(loc, action):
    initial = [0, 0]
    prev = list(loc)
    
    if (sum([abs(i) for i in action]) > 1):
        action = [sign(a) * .707 for a in action]

    loc[0] += action[0]
    loc[1] += action[1]

    if (is_terminal(loc[0], loc[1])):
        return loc, 1000, True
    
    # Check boundary conditions
    if (loc[0] < 0 or loc[0] > 9 or loc[1] < 0 or loc[1] > 9):
        return prev, -10, False

    # check cliff
    if (loc[0] >= 1 and loc[1] <= 9 and loc[1] >= 0 and loc[0] <= 1):
        return initial, -100, False

    return loc, -1, False

from itertools import permutations as perm
moves = list(perm([0, 1, -1], 2)) + [[1, 1], [-1, -1]]

def test_sim(init, moves):
    """
    Test simulator outcomes from initial state.
    Results are determined for each move in moves.
    """
    print(f"initial location: {init}")
    for m in moves:
        res, v, f = sim_environment(init, m)
        move = [round(i, 3) for i in m]
        res = [round(i, 3) for i in res]
        print(f"move: {move} ==> {res, v, f}")

        
# Some basic tests
inits = [[0,0],                                   # start
         [1,0],                                   # left of cliff
         [9,1],                                   # right of cliff
         [5,2],                                   # middle/above cliff
         [5,5],                                   # middle of grid
         [8,1],                                   # border terminal
         [9,2]                                    # border terminal
         ]
for i in inits:
    test_sim(i, moves)
