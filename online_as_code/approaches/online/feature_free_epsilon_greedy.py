import numpy as np
from numpy import ndarray
from approaches.online.bandit_selection_strategies.epsilon_greedy import EpsilonGreedy
import logging

logger = logging.getLogger("degroote")
logger.addHandler(logging.StreamHandler())

class FeatureFreeEpsilonGreedy:

    def __init__(self, bandit_selection_strategy=EpsilonGreedy(epsilon=0.05), imputation_strategy: str = 'cutoff'):
        self.bandit_selection_strategy = bandit_selection_strategy
        self.imputation_strategy = imputation_strategy

    def initialize(self, number_of_algorithms: int):
        self.performances_map = dict()
        self.number_of_algorithms = number_of_algorithms

        for algorithm_index in range(number_of_algorithms):
            self.performances_map[algorithm_index] = list()

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float, cutoff_time:float):
        if performance >= cutoff_time:
            if self.imputation_strategy == 'cutoff':
                performance = cutoff_time
            elif self.imputation_strategy == 'par10':
                performance = 10*cutoff_time
        self.performances_map[algorithm_id].append(performance)

    def predict(self, features: ndarray, instance_id: int, cutoff_time:float):
        predicted_performances = list()

        for algorithm_id in range(self.number_of_algorithms):
            average_performance = cutoff_time
            # if we do not have any values for this algorithm yet, pretend it always times out
            if len(self.performances_map[algorithm_id]) > 0:
                average_performance = np.mean(np.asarray(self.performances_map[algorithm_id]))
            predicted_performances.append(average_performance)

        return self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), None)

    def get_name(self):
        return 'feature_free_epsilon_greedy_' + self.imputation_strategy + "_epsilon=" + str(self.bandit_selection_strategy.epsilon)


