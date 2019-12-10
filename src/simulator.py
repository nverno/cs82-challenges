"""

"""
import numpy as np
import matplotlib.pyplot as plt

## -------------------------------------------------------------------
### Actions 

moves = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]

def vec_to_move(vec):
    v = np.array(vec, dtype=np.float)
    if (sum(abs(v)) > 1):
        v *= 0.707
    return v

def get_actions(actions=moves):
    """Convert actions to numpy arrays, and create reverse lookup tables."""
    vecs_normalized = list(map(lambda x: vec_to_move(x), actions))
    return vecs_normalized, {a: i for i,a in enumerate(actions)},\
        {i: a for i,a in enumerate(actions)}

def get_action_index(action):
    """Return index from action tuple."""
    return actions_index[action]

def get_index_action(index):
    """Return action tuple from index."""
    return actions_raw[index]

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
    prev = loc
    loc = loc + action

    # If hits a wall, stays in initial location
    if loc[0] < 0 or loc[0] > 9 or loc[1] < 0 or loc[1] > 9:
        return prev, -10, False

    # Returns to initial location if falls off of cliff
    if loc[0] >= 1 and loc[0] <= 9 and loc[0] <= 1:
        return np.array([0,0]), -100, False

    if is_terminal(loc):
        return loc, 1000, True
    
    return loc, -1, False
    
    
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
