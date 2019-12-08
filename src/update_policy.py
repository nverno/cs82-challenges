def update_policy(policy, Q, epsilon):
    '''
    Updates the policy based on estiamtes of Q using 
    an epslion greedy algorithm. The action with the highest
    action value is used.
    '''
    
    ## Find the keys for the actions in the policy
    keys = list(policy[0].keys())
    
    ## Iterate over the states and find the maximm action value.
    for state in range(len(policy)):
        ## First find the index of the max Q values  
        q = Q[state,:]
        max_action_index = np.where(q == max(q))[0]

        ## Find the probabilities for the transitions
        n_transitions = float(len(q))
        n_max_transitions = float(len(max_action_index))
        p_max_transitions = (1.0 - epsilon *(n_transitions - n_max_transitions))/(n_max_transitions)
        
        ## Now assign the probabilities to the policy as epsilon greedy.
        for key in keys:
            if(action_index[key] in max_action_index): policy[state][key] = p_max_transitions
            else: policy[state][key] = epsilon
    return(policy)                

update_policy(initial_policy, Q, 0.1)    
