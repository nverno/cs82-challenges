import numpy as np
from itertools import permutations as perm
moves = list(map(lambda x: np.array(x, dtype=np.float),
                 list(perm([0., 1., -1.], 2)) +
                 [(1., 1.), (-1., -1.)]))

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
         [8,1]                                    # border terminal
         ]
inits = map(lambda x: np.array(x, dtype=np.float), inits)

for i in inits:
    test_sim(i, moves)
