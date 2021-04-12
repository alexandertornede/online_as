from numpy import ndarray

class Template:
    def __init__(self):
        pass

    def initialize(self, number_of_algorithms: int, cutoff_time: float):
        pass

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float):
        pass

    def predict(self, features: ndarray, instance_id: int):
        pass