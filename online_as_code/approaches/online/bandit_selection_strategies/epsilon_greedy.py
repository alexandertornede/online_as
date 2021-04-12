from random import random
from numpy import ndarray
import numpy as np

class EpsilonGreedy:

    def __init__(self, epsilon: float, random_seed: int):
        self.epsilon = epsilon
        random.seed(random_seed)

    def select_based_on_predicted_performances(self, predicted_performances: ndarray):
        index_of_best_algorithm = np.argmin(predicted_performances)

        random_value = random.uniform(0,1)

        #initialize vector with 1 and set the value at the index of the selected algorithm to 0
        result_vector = np.ones(len(predicted_performances))
        result_vector.fill(1)

        if random_value < self.epsilon:
            #select any arm at random
            random_index = random.randint(0, len(predicted_performances)-1)
            result_vector[random_index] = 0
        else:
            #select best
            result_vector[index_of_best_algorithm] = 0
        return result_vector



