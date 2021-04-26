import numpy as np
from numpy import ndarray
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from approaches.online.bandit_selection_strategies.ucb import UCB
import math
import logging

logger = logging.getLogger("cox_regression")
logger.addHandler(logging.StreamHandler())

class OnlineLinearRegression:

    def __init__(self, bandit_selection_strategy):
        self.bandit_selection_strategy = bandit_selection_strategy
        self.all_training_samples = list()
        self.number_of_samples_seen = 0

    def initialize(self, number_of_algorithms: int, cutoff_time: float):
        self.number_of_algorithms = number_of_algorithms
        self.cutoff_time = cutoff_time
        self.current_A_map = None
        self.current_b_map = None
        self.data_for_algorithm = None
        self.maximum_feature_values = None
        self.minimum_feature_values = None
        self.mean_feature_values = None
        self.number_of_samples_seen = 0

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float):
        #initialize weight vectors randomly if not done yet
        if self.current_A_map is None:
            self.current_A_map = dict()
            self.current_b_map = dict()
            self.data_for_algorithm = dict()
            for algorithm_id in range(self.number_of_algorithms):
                self.current_A_map[algorithm_id] = np.identity(len(features))
                self.current_b_map[algorithm_id] = np.zeros(len(features))
                self.data_for_algorithm[algorithm_id] = False

            self.maximum_feature_values = np.full(features.size, -1000000)
            self.minimum_feature_values = np.full(features.size, +1000000)
            self.mean_feature_values = np.full(features.size, 0)

        self.data_for_algorithm[algorithm_id] = True

        imputed_sample = self.impute_sample(features)
        self.update_imputer(imputed_sample)

        self.update_scaler(imputed_sample)
        scaled_sample = self.scale_sample(imputed_sample)

        #TODO REWARD umshapen?
        reward = 1 - (performance / self.cutoff_time)

        self.current_A_map[algorithm_id] = np.add(self.current_A_map[algorithm_id], np.outer(scaled_sample, scaled_sample))
        self.current_b_map[algorithm_id] = self.current_b_map[algorithm_id] + reward * scaled_sample

    def update_imputer(self, sample: ndarray):
        #iteratively update the mean values of all features
        self.number_of_samples_seen+=1
        self.mean_feature_values = self.mean_feature_values + (1/self.number_of_samples_seen)*(sample - self.mean_feature_values)

    def impute_sample(self, sample: ndarray):
        #impute missing values
        mask = (np.isnan(sample))
        imputed_sample = np.copy(sample)
        imputed_sample[mask] = self.mean_feature_values[mask]
        return imputed_sample

    def update_scaler(self, sample: ndarray):
        self.minimum_feature_values = np.minimum(self.minimum_feature_values, sample)
        self.maximum_feature_values = np.maximum(self.maximum_feature_values, sample)

    def scale_sample(self, sample: ndarray):
        # min-max scaling
        max = 1
        min = 0
        # print("sample" + str(sample))
        # print(self.maximum_feature_values)
        # print(self.minimum_feature_values)
        # print((self.maximum_feature_values - self.minimum_feature_values))

        denominator = (self.maximum_feature_values - self.minimum_feature_values)

        #avoid division by zero => if denimonator is zero in one coordinate, X_std will be 0 anyway
        denominator[denominator == 0] = 1

        np.seterr(divide="raise", invalid="raise")

        X_std = ((sample - self.minimum_feature_values) / denominator)
        scaled_sample = X_std * (max - min) + min
        return np.clip(scaled_sample, a_min=0, a_max=1)

    def is_data_for_algorithm_present(self, algorithm_id):
        return self.data_for_algorithm is not None and self.data_for_algorithm[algorithm_id]

    def predict(self, features: ndarray, instance_id: int):
        logger.debug("instance_id: " + str(len(self.all_training_samples)))
        predicted_performances = list()
        confidence_bound_widths = list()
        if self.number_of_samples_seen >= 1:
            scaled_sample = self.impute_sample(features)
            scaled_sample = self.scale_sample(scaled_sample)

        for algorithm_id in range(self.number_of_algorithms):
            #if we have samples for that algorithms
            if self.is_data_for_algorithm_present(algorithm_id):
                #selection
                A_inv = np.linalg.inv(self.current_A_map[algorithm_id])
                theta_head = np.dot(A_inv, self.current_b_map[algorithm_id])
                estimated_reward = np.dot(theta_head, scaled_sample)
                confidence_bound_width = math.sqrt(np.linalg.multi_dot([scaled_sample, A_inv, scaled_sample])) # weight alpha is applied in UCB strategy
                predicted_performances.append(-estimated_reward)
                confidence_bound_widths.append(confidence_bound_width)

            else:
                #if not, set its performance to -100 such that it will get pulled for sure
                predicted_performances.append(-100)
                confidence_bound_widths.append(100000)

        logger.debug("pred_performances:" + str(predicted_performances))
        return self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), np.asarray(confidence_bound_widths))

    def get_name(self):
        name = 'online_linear_regression_{}'.format(type(self.bandit_selection_strategy).__name__)
        return name