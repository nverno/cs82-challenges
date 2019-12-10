"""
Functions to manage grid locations, visualization.
"""
import matplotlib.pyplot as plt
import numpy as np

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
    return num % 10, num // 10

## -------------------------------------------------------------------
### Visualize

def draw_grid(n):
    """Draw nxn grid."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # horizontal / vertical lines
    for ix in range(n):
        ax.plot([ix+1, ix+1], [0, n], linestyle='dashed', color='green')
    for ix in range(n-1):
        ax.plot([0, n], [ix+1, ix+1], linestyle='dashed', color='green')

    # boundaries
    ax.plot([0, 0], [0, n], color='black')
    ax.plot([0, n], [0, 0], color='black')
    ax.plot([n, 0], [n, n], color='black')
    ax.plot([n, n], [0, n], color='black')

    return fig, ax


def arrow_xy(action, prob):
    """Determine position change given action."""
    
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
