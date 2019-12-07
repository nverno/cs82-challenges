from math import cos
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
%matplotlib inline

def sim_environment(x, y, action):
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
            X_prime = x + 1
    elif (action[0] == -1):
        if (action[1] == 1):
            x_prime = x - .707
            y_prime = y + .707
        elif(action[1] == -1):
            x_prime = x - .707
            y_prime = y - .707
        else:
            X_prime = x - 1
    else:
        if (action[1] == 1):
            y_prime = y + 1
        elif (action[1] == -1):
            y_prime = y - 1
             
    #ensure new location is in bounds
    if (x_prime >= 10.0):
        x_prime = 0.0
        y_prime = 0.0
        hit_wall = True
    elif (x_prime <= 0.0):
        x_prime = 0.0
        y_prime = 0.0
        hit_wall = True
    if (y_prime >= 10.0):
        x_prime = 0.0
        y_prime = 0.0
        hit_wall = True
    elif (y_prime <= 0.0):
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
    elsif (fell):
        done = False
        reward = -100
    else if (hit_wall):
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
    if (y = 0.0 and x >= 9.0):
        return True
    elif (x = 10.0 and y <= 1.0):
        return True
    return False
