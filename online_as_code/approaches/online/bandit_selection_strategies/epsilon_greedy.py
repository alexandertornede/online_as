from random import random
from numpy import ndarray
import numpy as np

class EpsilonGreedy:

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_based_on_predicted_performances(self, predicted_performances: ndarray, confidence_interval_widths: ndarray):
        index_of_best_algorithm = np.argmin(predicted_performances)

        random_value = np.random.uniform()

        #initialize vector with 1 and set the value at the index of the selected algorithm to 0
        result_vector = np.ones(len(predicted_performances))
        result_vector.fill(1)

        if random_value < self.epsilon:
            #select any arm at random
            random_index = np.random.randint(low=0, high=len(predicted_performances))
            result_vector[random_index] = 0
        else:
            #select best
            result_vector[index_of_best_algorithm] = 0
        return result_vector