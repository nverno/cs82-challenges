#!/usr/bin/env python

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
