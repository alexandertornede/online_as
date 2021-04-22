import numpy as np
from numpy import ndarray

class StepFunction():

    def __init__(self, x: ndarray, y: ndarray):
        #assume x and accordingly to be sorted
        self.x = x
        self.y = y

    def get_value(self, x: float):
        i = 0
        while self.x[i] < x and i < self.x.size-1:
            i += 1

        # x == self.x[i]
        if x >= self.x[i]:
            return self.y[i]
        return self.y[i-1]