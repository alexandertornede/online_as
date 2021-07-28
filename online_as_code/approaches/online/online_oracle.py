import numpy as np
from numpy import ndarray

class OnlineOracle:

    def __init__(self):
        pass

    def initialize(self, number_of_algorithms: int):
        self.number_of_algorithms = number_of_algorithms

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float, cutoff_time:float):
        pass

    def predict(self, features: ndarray, instance_id: int, cutoff:float):
        #the online oracle assumes that the best algorithm id is given as the instance id and simply returns it
        result_vector = np.ones(self.number_of_algorithms)
        result_vector.fill(1)
        result_vector[instance_id] = 0
        return result_vector

    def get_name(self):
        return "online_oracle"