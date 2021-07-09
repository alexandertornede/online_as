import numpy as np
from numpy import ndarray
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from approaches.online.bandit_selection_strategies.ucb import UCB
import math
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from scipy.stats import halfnorm

logger = logging.getLogger("superset_lin_ucb")
logger.addHandler(logging.StreamHandler())

class SupersetConstrainedOptimization:

    def __init__(self, bandit_selection_strategy, alpha:float, C_tilde:float=2):
        self.bandit_selection_strategy = bandit_selection_strategy
        self.all_training_samples = list()
        self.number_of_samples_seen = 0
        self.lambda_map = None
        self.alpha = alpha
        self.C_tilde = C_tilde

    def initialize(self, number_of_algorithms: int):
        self.number_of_algorithms = number_of_algorithms
        self.current_A_map = None
        self.current_b_map = None
        self.current_f_map = None
        self.data_for_algorithm = None
        self.maximum_feature_values = None
        self.minimum_feature_values = None
        self.mean_feature_values = None
        self.number_of_algorithm_selections_with_timeout = None
        self.number_of_algorithm_selections = None
        self.number_of_samples_seen = 0

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float, cutoff_time:float):
        #initialize weight vectors randomly if not done yet
        if self.current_A_map is None:
            self.current_A_map = dict()
            self.current_b_map = dict()
            self.current_f_map = dict()
            self.lambda_map = dict()
            self.data_for_algorithm = dict()
            self.number_of_algorithm_selections_with_timeout = dict()
            self.number_of_algorithm_selections = dict()
            self.lr_classifier_map = dict()
            for algorithm_id in range(self.number_of_algorithms):
                self.current_A_map[algorithm_id] = np.identity(len(features))
                self.current_b_map[algorithm_id] = np.zeros(len(features))
                self.current_f_map[algorithm_id] = np.zeros(len(features))
                self.number_of_algorithm_selections_with_timeout[algorithm_id] = 0
                self.number_of_algorithm_selections[algorithm_id] = 0
                self.data_for_algorithm[algorithm_id] = False
                self.lambda_map[algorithm_id] = 0
                self.lr_classifier_map[algorithm_id] = SGDClassifier(loss='log')

            self.maximum_feature_values = np.full(features.size, -1000000)
            self.minimum_feature_values = np.full(features.size, +1000000)
            self.mean_feature_values = np.full(features.size, 0)

        self.data_for_algorithm[algorithm_id] = True


        imputed_sample = self.impute_sample(features)
        self.update_imputer(imputed_sample)

        self.update_scaler(imputed_sample)
        scaled_sample = self.scale_sample(imputed_sample)

        self.number_of_algorithm_selections[algorithm_id] = self.number_of_algorithm_selections[algorithm_id] + 1
        if performance >= cutoff_time:
            self.current_f_map[algorithm_id] = (self.number_of_algorithm_selections_with_timeout[algorithm_id] * self.current_f_map[algorithm_id] + scaled_sample) / (self.number_of_algorithm_selections_with_timeout[algorithm_id] + 1)

            self.number_of_algorithm_selections_with_timeout[algorithm_id] = self.number_of_algorithm_selections_with_timeout[algorithm_id] + 1
            performance = cutoff_time
            self.lr_classifier_map[algorithm_id].partial_fit(X=scaled_sample.reshape(1,-1), y=np.asarray([1]), classes=[0,1])
        else:
            self.lr_classifier_map[algorithm_id].partial_fit(X=scaled_sample.reshape(1,-1), y=np.asarray([0]), classes=[0,1])

        #lambda_param = self.number_of_algorithm_selections_with_timeout[algorithm_id] / (self.number_of_algorithm_selections[algorithm_id]+1)

        self.current_A_map[algorithm_id] = np.add(self.current_A_map[algorithm_id], np.outer(scaled_sample, scaled_sample))
        self.current_b_map[algorithm_id] = self.current_b_map[algorithm_id] + performance * scaled_sample
        A_inv = np.linalg.inv(self.current_A_map[algorithm_id])
        if 0 in self.current_f_map[algorithm_id]:
            self.lambda_map[algorithm_id] = 0
        else:
            self.lambda_map[algorithm_id] = 2*max(0, (cutoff_time - np.linalg.multi_dot([self.current_f_map[algorithm_id], A_inv, self.current_b_map[algorithm_id]]))/np.linalg.multi_dot([self.current_f_map[algorithm_id], A_inv, self.current_f_map[algorithm_id]]))

    def update_imputer(self, sample: ndarray):
        #iteratively update the mean values of all features
        self.number_of_samples_seen+=1
        self.mean_feature_values = self.mean_feature_values + (1/self.number_of_samples_seen)*(sample - self.mean_feature_values)

    def impute_sample(self, sample: ndarray):
        #impute missing values
        mask = (np.isnan(sample))
        imputed_sample = np.copy(sample)
        imputed_sample[mask] = self.mean_feature_values[mask]
        #if the sample contains only 0s as features, it can cause problems
        if np.all((imputed_sample == 0)):
            imputed_sample[0] = 0.000000001
        return imputed_sample

    def update_scaler(self, sample: ndarray):
        self.minimum_feature_values = np.minimum(self.minimum_feature_values, sample)
        self.maximum_feature_values = np.maximum(self.maximum_feature_values, sample)

    def scale_sample(self, sample: ndarray):

        # # min-max scaling
        # max = 1
        # min = 0
        # # print("sample" + str(sample))
        # # print(self.maximum_feature_values)
        # # print(self.minimum_feature_values)
        # # print((self.maximum_feature_values - self.minimum_feature_values))
        #
        # denominator = (self.maximum_feature_values - self.minimum_feature_values)
        #
        # #avoid division by zero => if denimonator is zero in one coordinate, X_std will be 0 anyway
        # denominator[denominator == 0] = 1
        #
        # np.seterr(divide="raise", invalid="raise")
        #
        # X_std = ((sample - self.minimum_feature_values) / denominator)
        # scaled_sample = X_std * (max - min) + min
        # return np.clip(scaled_sample, a_min=0, a_max=1)
        scaled_sample = sample/np.linalg.norm(sample)
        return scaled_sample

    def is_data_for_algorithm_present(self, algorithm_id):
        return self.data_for_algorithm is not None and self.data_for_algorithm[algorithm_id]

    def predict(self, features: ndarray, instance_id: int, cutoff_time: float):
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
                b = self.current_b_map[algorithm_id]
                theta_a = np.dot(A_inv, (b + self.lambda_map[algorithm_id]*self.current_f_map[algorithm_id]))

                s = math.sqrt(np.linalg.multi_dot([scaled_sample, A_inv, scaled_sample])) * math.sqrt(self.number_of_algorithm_selections_with_timeout[algorithm_id]) * cutoff_time

                predicted_performance = np.dot(theta_a, scaled_sample)
                bound = self.alpha * s * halfnorm.rvs(loc=0, scale=0.25)
                #probability_term = (10*cutoff_time - np.dot(theta_a, scaled_sample))*((np.dot(theta_a, scaled_sample) - self.alpha * s - min(np.dot(theta_a, scaled_sample), cutoff_time))/(self.C_tilde * cutoff_time))
                #probability_term = max(0,1 - (cutoff_time - min(max(0, predicted_performance), cutoff_time))/cutoff_time) #self.number_of_algorithm_selections_with_timeout[algorithm_id] / (self.number_of_algorithm_selections[algorithm_id]+1)
                probability_term = self.lr_classifier_map[algorithm_id].predict_proba(X=scaled_sample.reshape(1,-1))[0][1]

                l_a = predicted_performance - bound + (10*cutoff_time - predicted_performance) * probability_term


                predicted_performances.append(l_a)
                confidence_bound_widths.append(0)

            else:
                #if not, set its performance to -100 such that it will get pulled for sure
                predicted_performances.append(-1000000)
                confidence_bound_widths.append(100000)

        logger.debug("pred_performances:" + str(predicted_performances))
        logger.debug("confidence_bound_withdts: " + str(confidence_bound_widths))
        final_prediction_vector = self.bandit_selection_strategy.select_based_on_predicted_performances(np.asarray(predicted_performances), np.asarray(confidence_bound_widths))

        selected_algorithm_id = np.argmin(final_prediction_vector)

        return final_prediction_vector

    def get_name(self):
        name = 'super_set_co_Ctilde={}_{}'.format(self.C_tilde, type(self.bandit_selection_strategy).__name__)
        return name