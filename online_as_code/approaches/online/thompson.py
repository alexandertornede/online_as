import numpy as np
from numpy import ndarray
import math
import logging
from scipy.stats import norm

logger = logging.getLogger("thompson")
logger.addHandler(logging.StreamHandler())

class Thompson:

    def __init__(self, sigma:float, lamda:float, revisited: bool, true_expected_value: bool, buckley_james: bool = False):
        self.all_training_samples = list()
        self.number_of_samples_seen = 0
        self.sigma = sigma
        self.lamda = lamda
        self.buckley_james = buckley_james
        self.revisited = revisited
        self.true_expected_value = true_expected_value

    def initialize(self, number_of_algorithms: int):
        self.number_of_algorithms = number_of_algorithms
        self.current_b_map = None
        self.current_A_map = None
        self.data_for_algorithm = None
        self.mean_feature_values = None
        self.number_of_algorithm_selections_with_timeout = None
        self.number_of_algorithm_selections = None
        self.number_of_samples_seen = 0

    def train_with_single_instance(self, features: ndarray, algorithm_id: int, performance: float, cutoff_time:float):
        #initialize weight vectors randomly if not done yet
        if self.current_A_map is None:
            self.current_b_map = dict()
            self.current_A_map = dict()
            self.data_for_algorithm = dict()
            self.number_of_algorithm_selections_with_timeout = dict()
            self.number_of_algorithm_selections = dict()
            for algorithm_id in range(self.number_of_algorithms):
                self.current_b_map[algorithm_id] = np.zeros(len(features))
                self.current_A_map[algorithm_id] = np.identity(len(features))*self.lamda
                self.number_of_algorithm_selections_with_timeout[algorithm_id] = 0
                self.number_of_algorithm_selections[algorithm_id] = 0
                self.data_for_algorithm[algorithm_id] = False

            self.mean_feature_values = np.full(features.size, 0)

        self.data_for_algorithm[algorithm_id] = True

        imputed_sample = self.impute_sample(features)
        self.update_imputer(imputed_sample)

        scaled_sample = self.scale_sample(imputed_sample)

        self.number_of_algorithm_selections[algorithm_id] = self.number_of_algorithm_selections[algorithm_id] + 1
        if performance >= cutoff_time:
            self.number_of_algorithm_selections_with_timeout[algorithm_id] = self.number_of_algorithm_selections_with_timeout[algorithm_id] + 1
            if self.buckley_james:
                A_inv = np.linalg.inv(self.current_A_map[algorithm_id])
                b = self.current_b_map[algorithm_id]
                theta_a = np.dot(A_inv, b)

                sample_theta_based_performance = 0
                counter = 0
                while sample_theta_based_performance <= math.log(cutoff_time) and counter < 20:
                    sampled_theta = np.random.multivariate_normal(mean=theta_a, cov=self.sigma*A_inv)
                    sample_theta_based_performance = np.dot(scaled_sample, sampled_theta)
                    counter += 1
                performance_to_use_for_update = sample_theta_based_performance
                if counter >= 20:
                    performance_to_use_for_update = math.log(cutoff_time)
            else:
                performance_to_use_for_update = math.log(cutoff_time)
        else:
            if performance == 0:
                performance = 0.0001
            performance_to_use_for_update = math.log(performance)

        self.current_b_map[algorithm_id] = self.current_b_map[algorithm_id] + performance_to_use_for_update * scaled_sample
        self.current_A_map[algorithm_id] = self.current_A_map[algorithm_id] + np.outer(scaled_sample, scaled_sample)

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

    def scale_sample(self, sample: ndarray):
        return sample / np.linalg.norm(sample)

    def is_data_for_algorithm_present(self, algorithm_id):
        return self.data_for_algorithm is not None and self.data_for_algorithm[algorithm_id]

    def predict(self, features: ndarray, instance_id: int, cutoff:float):
        logger.debug("instance_id: " + str(len(self.all_training_samples)))
        predicted_performances = list()
        confidence_bound_widths = list()
        if self.number_of_samples_seen >= 1:
            scaled_sample = self.impute_sample(features)
            scaled_sample = self.scale_sample(scaled_sample)

        # cdfs_debug = list()
        # sample_based_performance_debug = list()
        # scale_debug = list()
        for algorithm_id in range(self.number_of_algorithms):
            #if we have samples for that algorithms
            if self.is_data_for_algorithm_present(algorithm_id):
                #selection
                A_inv = np.linalg.inv(self.current_A_map[algorithm_id])
                b = self.current_b_map[algorithm_id]
                theta_a = np.dot(A_inv, b)

                sampled_theta = np.random.multivariate_normal(mean=theta_a, cov=self.sigma*A_inv)
                sample_theta_based_performance = np.dot(scaled_sample, sampled_theta)
                # sample_based_performance_debug.append(sample_theta_based_performance)
                if self.revisited:
                    scale = self.sigma*np.linalg.multi_dot([scaled_sample, self.sigma*A_inv, scaled_sample])
                    # scale_debug.append(scale)
                    cdf = norm.cdf(x=math.log(cutoff), loc=sample_theta_based_performance, scale=scale)
                    # cdfs_debug.append(cdf)
                    if self.true_expected_value:
                        C_tilde_1 = (math.log(cutoff) - sample_theta_based_performance - scale/2) / math.sqrt(scale)
                        C_tilde_2 = (math.log(cutoff) - sample_theta_based_performance) / math.sqrt(scale)
                        inverse_mills_ratio = norm.cdf(loc=0, scale=1, x=C_tilde_1) / norm.cdf(loc=0, scale=1, x=C_tilde_2)
                        E_tilde_C = math.exp(sample_theta_based_performance + scale/2) * inverse_mills_ratio
                        l_a = E_tilde_C + (1 - cdf) * (math.log(10*cutoff) - E_tilde_C)
                    else:
                        C_tilde = (math.log(cutoff) - sample_theta_based_performance) / math.sqrt(scale)
                        inverse_mills_ratio = norm.pdf(loc=0, scale=1, x=C_tilde) / norm.cdf(loc=0, scale=1, x=C_tilde)
                        l_a = sample_theta_based_performance + (1 - cdf) * (math.log(10*cutoff) - sample_theta_based_performance + math.sqrt(scale) * inverse_mills_ratio )
                else:
                    l_a = sample_theta_based_performance

                predicted_performances.append(l_a)

            else:
                #if not, set its performance to -100 such that it will get pulled for sure
                predicted_performances.append(-1000000)

        #print("cdf: " + str(cdfs_debug))
        #print("perf: " + str(sample_based_performance_debug))
        #print("scale: " + str(scale_debug))
        logger.debug("pred_performances:" + str(predicted_performances))

        return np.asarray(predicted_performances)

    def get_name(self):
        name = 'thompson'
        if self.true_expected_value:
            name = 'e_' + name
        if self.revisited:
            name = name + "_rev"
        if self.buckley_james:
            name = 'bj_' + name
        name = name + '_sigma={}_lambda={}'.format(str(self.sigma), str(self.lamda))
        return name