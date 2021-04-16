import numpy as np
from numpy import ndarray
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import forestci as fci
from approaches.online.bandit_selection_strategies.ucb import UCB
import math
import logging

logger = logging.getLogger("cox_regression")
logger.addHandler(logging.StreamHandler())

class CoxRegression:

    def __init__(self, bandit_selection_strategy, learning_rate: float = 0.001):
        self.raw_preprocessing_pipeline = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
        self.trained_preprocessing_pipeline = clone(self.raw_preprocessing_pipeline)
        self.bandit_selection_strategy = bandit_selection_strategy
        self.learning_rate = learning_rate

    def initialize(self, number_of_algorithms: int, cutoff_time: float):
        self.number_of_algorithms = number_of_algorithms
        self.cutoff_time = cutoff_time
        self.current_training_X_map = dict()
        self.current_training_y_map = dict()
        self.current_weight_map = None

        self.trained_preprocessing_pipeline = clone(self.raw_preprocessing_pipeline)

        for algorithm_index in range(number_of_algorithms):
            self.current_training_X_map[algorithm_index] = list()
            self.current_training_y_map[algorithm_index] = list()

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float):
        #initialize weight vectors randomly if not done yet
        if self.current_weight_map is None:
            self.current_weight_map = dict()
            for algorithm_id in range(self.number_of_algorithms):
                self.current_weight_map[algorithm_id] = np.random.rand(len(features))

        #store new training sample
        self.current_training_X_map[algorithm_id].append(features)
        self.current_training_y_map[algorithm_id].append(performance)

        #retrain preprocessing pipeline according with old data augmented by new sample
        self.trained_preprocessing_pipeline = clone(self.raw_preprocessing_pipeline)
        X_train = np.asarray(self.current_training_X_map[algorithm_id])
        self.trained_preprocessing_pipeline.fit(X_train)

        #run feature vector of new sample through preprocessing
        new_sample = self.trained_preprocessing_pipeline.transform(np.reshape(features,
                                                                              (1, len(features))))
        is_censored_sample = self.is_time_censored(performance)

        #obtain weight vector of algorithm to update
        weight_vector_to_update = self.current_weight_map[algorithm_id]


        #perform weight update
        #note that we only need to update the weights if the sample does not feature a timeout, otherwise we only need to update the risk sets
        if not is_censored_sample:
            #compute gradient
            risk_set = self.compute_risk_set_for_instance_in_algorithm_dataset(algorithm_id, performance)
            denominator = np.asarray(list(map(lambda instance: math.exp(np.dot(instance, weight_vector_to_update)), risk_set)))
            nominator = np.asarray(list(map(lambda instance:  math.exp(np.dot(instance, weight_vector_to_update)) * instance, risk_set)))
            gradient = -(new_sample - nominator/denominator)

            #perform gradient step
            self.current_weight_map[algorithm_id] = weight_vector_to_update - self.learning_rate*gradient


    def compute_baseline_survival_function(self, algorithm_id: int, timestep: float):
        y = self.current_training_y_map[algorithm_id]
        weight_vector = self.current_weight_map[algorithm_id]

        sum = 0
        for observed_time in y:
            if not self.is_time_censored(observed_time) and min(observed_time, self.cutoff_time) <= timestep:
                risk_set = self.compute_risk_set_for_instance_in_algorithm_dataset(algorithm_id, observed_time)
                denominator = np.asarray(list(map(lambda instance: math.exp(np.dot(instance, weight_vector)), risk_set)))
                sum += (1/denominator)
        return sum

    def compute_survival_function(self, algorithm_id: int, instance: ndarray, timestep: float):
        weight_vector = self.current_weight_map[algorithm_id]
        return math.pow(self.compute_baseline_survival_function(algorithm_id=algorithm_id, timestep=timestep), math.exp(np.dot(weight_vector, np.reshape(instance, len(weight_vector)))))


    def compute_expected_par10_time(self, algorithm_id: int, instance:ndarray):
        y = self.current_training_y_map[algorithm_id].copy()
        y.append(0)
        y.append(self.cutoff_time)
        y = list(set(self.current_training_y_map[algorithm_id]))

        times = np.sort(np.asarray(y))
        sum = 0
        for i in range(len(times) -1):
            time = times[i]
            next_time = times[i+1]
            sum += time*(self.compute_survival_function(algorithm_id=algorithm_id, instance=instance, timestep = time) - self.compute_survival_function(algorithm_id=algorithm_id, instance=instance, timestep = next_time))
        sum += 10*self.compute_survival_function(algorithm_id=algorithm_id, instance=instance, timestep = self.cutoff_time)
        return sum

    def is_time_censored(self, time:float):
        return time >= self.cutoff_time

    def compute_risk_set_for_instance_in_algorithm_dataset(self, algorithm_id: int, performance:float):
        X = self.current_training_X_map[algorithm_id]
        y = self.current_training_y_map[algorithm_id]

        indices_of_risk_set_elements = np.argwhere(min(np.asarray(y), np.full(len(y), self.cutoff_time)) >= min(performance, self.cutoff_time))[0]

        risk_set = list()
        for i in indices_of_risk_set_elements:
            risk_set.append(self.trained_preprocessing_pipeline.transform(np.reshape(X[i],(1,len(X[i])))))

        return risk_set

    def is_data_for_algorithm_present(self, algorithm_id):
        return len(self.current_training_X_map[algorithm_id]) > 0

    def predict(self, features: ndarray, instance_id: int):
        predicted_performances = list()
        current_standard_deviation = list()

        if self.current_weight_map is not None:
            preprocessed_instance = self.trained_preprocessing_pipeline.transform(np.reshape(features,
                                                                                             (1, len(features))))

            for algorithm_id in range(self.number_of_algorithms):
                predicted_performances.append(self.compute_expected_par10_time(algorithm_id=algorithm_id, instance = preprocessed_instance))
        else:
            for algorithm_id in range(self.number_of_algorithms):
                predicted_performances.append(0)
        return self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), np.asarray(current_standard_deviation))

    def get_name(self):
        name = 'cox_regression_{}'.format(type(self.bandit_selection_strategy).__name__)
        return name