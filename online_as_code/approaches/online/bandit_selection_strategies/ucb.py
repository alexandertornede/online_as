from numpy import ndarray
import numpy as np

class UCB:

    def __init__(self):
        pass

    def select_based_on_predicted_performances(self, predicted_performances: ndarray, confidence_interval_widths: ndarray):
        index_of_best_algorithm = np.argmin(np.add(predicted_performances, confidence_interval_widths))

        #initialize vector with 1 and set the value at the index of the selected algorithm to 0
        result_vector = np.ones(len(predicted_performances))
        result_vector.fill(1)

        result_vector[index_of_best_algorithm] = 0

        return result_vector