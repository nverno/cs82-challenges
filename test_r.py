#!/usr/bin/env python


from numpy.random import choice
import timeit

timeit.timeit("choice(1)", number=10000)
