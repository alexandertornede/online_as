import numpy as np
from numpy import ndarray
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import forestci as fci
from approaches.online.bandit_selection_strategies.ucb import UCB
import math
import logging
from approaches.online.step_function import StepFunction

import matplotlib.pyplot as plt

logger = logging.getLogger("cox_regression")
# logger.addHandler(logging.StreamHandler())

class CoxRegression:

    def __init__(self, bandit_selection_strategy, learning_rate: float = 0.01, regularization_parameter: float = 0, window_size: float = 50):
        self.bandit_selection_strategy = bandit_selection_strategy
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.all_training_samples = list()
        self.number_of_instances_seen = 0
        self.window_size = window_size

    def initialize(self, number_of_algorithms: int, cutoff_time: float):
        #for debugging purposes
        np.seterr(divide="raise", invalid="raise")

        self.number_of_algorithms = number_of_algorithms
        self.cutoff_time = cutoff_time
        self.all_training_samples = list()
        self.current_training_X_transformed_map = dict()
        self.current_training_y_map = dict()
        self.current_weight_map = None
        self.precomputed_baseline_survival_functions = dict()
        self.number_of_instances_seen = 0

        for algorithm_index in range(number_of_algorithms):
            self.current_training_y_map[algorithm_index] = list()
            self.current_training_X_transformed_map[algorithm_index] = list()

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float):
        self.number_of_instances_seen += 1

        if not np.isnan(features).any():
            self.found_non_nan_training_sample = True

        #initialize weight vectors randomly if not done yet
        if self.current_weight_map is None:
            self.current_weight_map = dict()
            for algorithm_id in range(self.number_of_algorithms):
                self.current_weight_map[algorithm_id] = np.random.rand(len(features)+1) # +1 for bias

            self.maximum_feature_values = np.full(features.size, -1000000)
            self.minimum_feature_values = np.full(features.size, +1000000)
            self.mean_feature_values = np.full(features.size, 0)


        imputed_sample = self.impute_sample(features)
        self.update_imputer(imputed_sample)

        self.update_scaler(imputed_sample)
        scaled_sample = self.scale_sample(imputed_sample)

        scaled_sample = np.append(scaled_sample, values=[1])
        self.add_sample(algorithm_id=algorithm_id, features=scaled_sample, runtime=performance)

        #run feature vector of new sample through preprocessing
        is_censored_sample = self.is_time_censored(performance)

        #obtain weight vector of algorithm to update
        weight_vector_to_update = self.current_weight_map[algorithm_id]

        #note that we only need to update the weights if the sample does not feature a timeout, otherwise we only need to update the risk sets
        if not is_censored_sample:
            #compute gradient
            risk_set = self.compute_risk_set_for_instance_in_algorithm_dataset(algorithm_id, performance)
            logger.debug("w: " + str(weight_vector_to_update))
            logger.debug("risk_set: " + str(risk_set))
            denominator = np.sum(np.asarray(list(map(lambda instance: math.exp(self.scalar_product(instance, weight_vector_to_update)), risk_set))))
            nominator = np.sum(np.asarray(list(map(lambda instance:  math.exp(self.scalar_product(instance, weight_vector_to_update)) * instance, risk_set))), axis=0).flatten()
            gradient = -(scaled_sample - nominator/denominator)
            logger.debug("gradient: " + str(algorithm_id) + ": " + str(gradient))

            #perform gradient step
            self.current_weight_map[algorithm_id] = (weight_vector_to_update - self.learning_rate*gradient)
            logger.debug("update: " + str(algorithm_id) + ": " + str(self.current_weight_map[algorithm_id]))

        self.precompute_baseline_survival_function(algorithm_id=algorithm_id)

    def add_sample(self, algorithm_id: int, features: ndarray, runtime: float):
        self.current_training_X_transformed_map[algorithm_id].append(features)
        self.current_training_y_map[algorithm_id].append(runtime)
        assert len(self.current_training_X_transformed_map[algorithm_id]) == len(self.current_training_y_map[algorithm_id])

        #make sure, we do not store too many samples for each algorithm
        if self.window_size > 0:
            if len(self.current_training_X_transformed_map[algorithm_id]) > self.window_size:
                self.current_training_X_transformed_map[algorithm_id].pop(0)
                self.current_training_y_map[algorithm_id].pop(0)

    def update_imputer(self, sample: ndarray):
        #iteratively update the mean values of all features
        self.number_of_instances_seen+=1
        self.mean_feature_values = self.mean_feature_values + (1/self.number_of_instances_seen)*(sample - self.mean_feature_values)

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

        denominator = (self.maximum_feature_values - self.minimum_feature_values)

        #avoid division by zero => if denimonator is zero in one coordinate, X_std will be 0 anyway
        denominator[denominator == 0] = 1

        X_std = ((sample - self.minimum_feature_values) / denominator)
        scaled_sample = X_std * (max - min) + min
        return np.clip(scaled_sample, a_min=0, a_max=1)

    def precompute_baseline_survival_function(self, algorithm_id:int):
        y = self.current_training_y_map[algorithm_id].copy()
        y.append(0)
        y.append(self.cutoff_time)
        y = list(set(y))

        times = np.sort(np.asarray(y))
        survival_times = np.asarray(list(map(lambda time: self.compute_baseline_survival_function(algorithm_id=algorithm_id, timestep=time, use_precomputed_function=False), times)))
        self.precomputed_baseline_survival_functions[algorithm_id] = StepFunction(times, survival_times)

        # if self.number_of_instances_seen == 200:
        #     for i in range(0, self.number_of_algorithms):
        #         plt.ylabel("Baseline survival function: Algorithm " + str(i))
        #         plt.xlabel("Runtime")
        #         #plt.ylim(ymax = 10, ymin= 0)
        #         plt.legend()
        #         plt.grid(True)
        #         y = self.current_training_y_map[i].copy()
        #         y.append(0)
        #         y.append(self.cutoff_time)
        #
        #         for y_t in y:
        #             value = self.precomputed_baseline_survival_functions[i].get_value(y_t)
        #             print(value)
        #             plt.scatter(y_t, value)
        #         plt.show()


    def compute_risk_set_for_instance_in_algorithm_dataset(self, algorithm_id: int, performance:float):
        X = self.current_training_X_transformed_map[algorithm_id]
        y = np.asarray(self.current_training_y_map[algorithm_id])
        # y = np.where(y > self.cutoff_time, self.cutoff_time, y)

        mask_for_risk_set_elements = y >= min(performance, self.cutoff_time)
        risk_set = list()
        for index, value in enumerate(mask_for_risk_set_elements):
            if value:
                risk_set.append(X[index])

        if len(risk_set) == 0:
            logger.error("risk set is empty")
        return risk_set

    def scalar_product(self, vector1: ndarray, vector2: ndarray):
        scalar_product = np.dot(vector1, vector2)
        if scalar_product == 0:
            return 0
        return scalar_product #/(np.linalg.norm(vector1)*np.linalg.norm(vector2))


    def compute_baseline_survival_function(self, algorithm_id: int, timestep: float, use_precomputed_function:bool = True):
        if use_precomputed_function:
            precomputed_value = self.precomputed_baseline_survival_functions[algorithm_id].get_value(timestep)
            return precomputed_value

        y = self.current_training_y_map[algorithm_id]
        weight_vector = self.current_weight_map[algorithm_id]

        sum = 0
        for observed_time in y:
            if not self.is_time_censored(observed_time) and min(observed_time, self.cutoff_time) <= timestep:
                risk_set = self.compute_risk_set_for_instance_in_algorithm_dataset(algorithm_id, observed_time)
                denominator = np.sum(np.asarray(list(map(lambda instance: math.exp(self.scalar_product(instance, weight_vector)), risk_set))))
                sum += (1/denominator)
        return math.exp(-sum)

    def compute_survival_function(self, algorithm_id: int, instance: ndarray, timestep: float):
        weight_vector = self.current_weight_map[algorithm_id]
        # baseline_survival_function_value = self.compute_baseline_survival_function(algorithm_id=algorithm_id, timestep=timestep) #TODO
        baseline_survival_function_value = self.compute_baseline_survival_function(algorithm_id=algorithm_id, timestep=timestep)
        weight_vector_instance_dot_product = self.scalar_product(weight_vector, instance)
        exponent = math.exp(weight_vector_instance_dot_product)
        result = math.pow(baseline_survival_function_value, exponent)
        if result > 1 or result < 0:
            logger.error("fail")
        return result

    def compute_expected_par10_time(self, algorithm_id: int, instance:ndarray):
        y = self.current_training_y_map[algorithm_id].copy()
        y.append(0)
        y.append(self.cutoff_time)
        y = list(set(y))

        times = np.sort(np.asarray(y))
        survival_times = np.asarray(list(map(lambda time: self.compute_survival_function(algorithm_id=algorithm_id, instance = instance, timestep=time), times)))
        precomputed_survival_function = StepFunction(times, survival_times)

        sum = 0
        for i in range(len(times) -1):
            time = times[i]
            next_time = times[i+1]
            #sum += time*(self.compute_survival_function(algorithm_id=algorithm_id, instance=instance, timestep = time) - self.compute_survival_function(algorithm_id=algorithm_id, instance=instance, timestep = next_time))
            sum += time*(precomputed_survival_function.get_value(time) - precomputed_survival_function.get_value(next_time))
        #sum += 10*self.compute_survival_function(algorithm_id=algorithm_id, instance=instance, timestep = self.cutoff_time)
        sum += 10*precomputed_survival_function.get_value(self.cutoff_time)
        return sum

    def is_time_censored(self, time:float):
        return time >= self.cutoff_time

    def is_data_for_algorithm_present(self, algorithm_id):
        return len(self.current_training_X_transformed_map[algorithm_id]) > 0

    def predict(self, features: ndarray, instance_id: int):
        predicted_performances = list()
        current_standard_deviation = list()

        if self.current_weight_map is not None:
            preprocessed_instance = self.scale_sample(self.impute_sample(features))
            preprocessed_instance = np.append(preprocessed_instance, values=[1])
            for algorithm_id in range(self.number_of_algorithms):
                #if we have samples for that algorithms
                if self.is_data_for_algorithm_present(algorithm_id):
                    predicted_performances.append(self.compute_expected_par10_time(algorithm_id=algorithm_id, instance=preprocessed_instance))
                else:
                    #if not, set its performance to -1 such that it will get pulled for sure
                    predicted_performances.append(-1)
        else:
            for algorithm_id in range(self.number_of_algorithms):
                predicted_performances.append(-1)

        logger.info("instance_id: " + str(instance_id) + " - pred_performances:" + str(predicted_performances))
        return self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), np.asarray(current_standard_deviation))

    def get_name(self):
        name = 'cox_regression_{}'.format(type(self.bandit_selection_strategy).__name__)
        return name