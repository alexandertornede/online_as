import numpy as np
from numpy import ndarray
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import forestci as fci
from approaches.online.bandit_selection_strategies.ucb import UCB
import math
import logging

logger = logging.getLogger("degroote")
logger.addHandler(logging.StreamHandler())

class Degroote:

    def __init__(self, bandit_selection_strategy, regression_model=RandomForestRegressor(n_jobs=1, n_estimators=100), standard_deviation_lower_bound=1.0):
        self.pure_regression_model = regression_model
        self.regression_model = Pipeline([('imputer', SimpleImputer()), ('scaler', MinMaxScaler(clip=True)), ('model', regression_model)])
        self.bandit_selection_strategy = bandit_selection_strategy
        self.standard_deviation_lower_bound = standard_deviation_lower_bound

    def initialize(self, number_of_algorithms: int):
        self.number_of_algorithms = number_of_algorithms
        self.current_training_X_map = dict()
        self.current_training_y_map = dict()
        self.trained_models_map = dict()
        self.non_nan_training_sample_checking_map = dict()

        for algorithm_index in range(number_of_algorithms):
            self.trained_models_map[algorithm_index] = clone(self.regression_model)
            self.current_training_X_map[algorithm_index] = list()
            self.current_training_y_map[algorithm_index] = list()
            self.non_nan_training_sample_checking_map[algorithm_index] = False

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float, cutoff_time: float):
        #store new training sample
        self.current_training_X_map[algorithm_id].append(features)
        self.current_training_y_map[algorithm_id].append(performance)

        if not np.isnan(features).any():
            self.non_nan_training_sample_checking_map[algorithm_id] = True

        # only (re)train the model if we have seen at least one instance with non nan values as the imputer does not work otherwise
        if self.non_nan_training_sample_checking_map[algorithm_id]:
            #retrain according model
            self.trained_models_map[algorithm_id] = clone(self.regression_model)

            X_train = np.asarray(self.current_training_X_map[algorithm_id])
            y_train = np.asarray(self.current_training_y_map[algorithm_id])
            self.trained_models_map[algorithm_id] = clone(self.regression_model)
            self.trained_models_map[algorithm_id].fit(X_train, y_train)

    def is_data_for_algorithm_present(self, algorithm_id):
        return len(self.current_training_X_map[algorithm_id]) > 0 and self.non_nan_training_sample_checking_map[algorithm_id]

    def predict(self, features: ndarray, instance_id: int, cutoff_time: float):
        predicted_performances = list()
        current_standard_deviation = list()

        for algorithm_id in range(self.number_of_algorithms):
            # if the model is not fitted yet, just predict the cutoff time
            prediction = -100000
            standard_deviation = 1
            if self.is_data_for_algorithm_present(algorithm_id):
                X_test = np.reshape(features,
                                    (1, len(features)))

                regression_pipeline = self.trained_models_map[algorithm_id]
                prediction = regression_pipeline.predict(X_test)[0]
                logger.debug("prediction: " + str(prediction))

                if isinstance(self.bandit_selection_strategy, UCB):
                    #Determine standard deviation using the jack knife method
                    regressor_for_algorithm_id = regression_pipeline['model']
                    X_train = np.asarray(self.current_training_X_map[algorithm_id])
                    X_train = regression_pipeline['imputer'].transform(X_train)
                    X_train = regression_pipeline['scaler'].transform(X_train)

                    X_test = regression_pipeline['imputer'].transform(X_test)
                    X_test = regression_pipeline['scaler'].transform(X_test)

                    X_test_for_standard_deviation = np.vstack([X_train, X_test])

                    # we need at least 3 samples such that the package works correctly
                    if len(X_test_for_standard_deviation) > 2:
                        variance = fci.random_forest_error(forest=regressor_for_algorithm_id, X_train=X_train, X_test=X_test_for_standard_deviation, calibrate=False)
                        relevant_variance = variance[len(variance)-1]
                        logger.debug("std: " + str(relevant_variance))
                    else:
                        relevant_variance = self.standard_deviation_lower_bound

                    #TODO How to deal with negative standard error?
                    if not math.isnan(relevant_variance):
                        relevant_variance = max([self.standard_deviation_lower_bound, relevant_variance])
                        standard_deviation = math.sqrt(relevant_variance)

            current_standard_deviation.append(standard_deviation)
            predicted_performances.append(prediction)

        return self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), np.asarray(current_standard_deviation))

    def get_name(self):
        name = 'degroote_{}_{}'.format(type(self.bandit_selection_strategy).__name__,type(self.pure_regression_model).__name__)
        return name


