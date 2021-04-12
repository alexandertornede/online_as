import numpy as np
from numpy import ndarray
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import forestci as fci


class Degroote:

    #TODO add actual selection strategy
    def __init__(self, bandit_selection_strategy, regression_model=RandomForestRegressor(n_jobs=1, n_estimators=100)):
        self.regression_model = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer()), ('model', regression_model)])
        self.bandit_selection_strategy = bandit_selection_strategy

    def initialize(self, number_of_algorithms: int, cutoff_time: float):
        self.number_of_algorithms = number_of_algorithms
        self.cutoff_time = cutoff_time
        self.current_training_X_map = dict()
        self.current_training_y_map = dict()

        self.current_standard_deviation = np.full(number_of_algorithms, -100000.0)

        for algorithm_index in range(number_of_algorithms):
            self.trained_models[algorithm_index] = self.regression_model = clone(self.regression_model)
            self.current_training_X_map[algorithm_index] = list()
            self.current_training_y_map[algorithm_index] = list()
             #initialize really low such each arm will be pulled at least once

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float):
        #store new training sample
        self.current_training_X_map[algorithm_id].append(features)
        self.current_training_y_map[algorithm_id].append(performance)

        #retrain models
        self.trained_models[algorithm_id] = clone(self.regression_model)
        if len(self.current_training_X_map[algorithm_id]) > 0:
            X_train = np.asarray(self.current_training_X_map[algorithm_id])
            y_train = np.asarray(self.current_training_y_map[algorithm_id])
            self.trained_models[algorithm_id].fit(X_train, y_train)

            #Determine standard deviation using the jack knife method
            self.current_standard_deviation_map[algorithm_id] = fci.random_forest_error(self.trained_models[algorithm_id], X_train, X_train)


    def predict(self, features: ndarray, instance_id: int):
        predicted_performances = list()

        for algorithm_id in range(self.num_algorithms):
            X_test = np.reshape(features,
                                (1, len(features)))

            model = self.trained_models[algorithm_id]

            prediction = model.predict(X_test)
            predicted_performances.append(prediction)

        return self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), self.current_standard_deviation)


