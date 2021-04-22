import copy
import logging
import numpy as np
import os
import time
from aslib_scenario.aslib_scenario import ASlibScenario
from simple_runtime_metric import RuntimeMetric
from numpy import ndarray

logger = logging.getLogger("evaluate_train_test_split")
logger.addHandler(logging.StreamHandler())


def evaluate_train_test_split(scenario: ASlibScenario, approach, metrics, fold: int, amount_of_training_instances: int, censored_value_imputation:str):

    if censored_value_imputation != 'all':
        scenario = copy.deepcopy(scenario)
        threshold = scenario.algorithm_cutoff_time
        if censored_value_imputation == 'clip_censored':
            scenario.performance_data = scenario.performance_data.clip(upper=threshold)

        elif censored_value_imputation == 'ignore_censored':
            scenario.performance_data = scenario.performance_data.replace(10*threshold, np.nan)

    approach.initialize(len(scenario.algorithms), scenario.algorithm_cutoff_time)

    approach_metric_values = np.zeros(len(metrics))

    num_counted_test_values = 0

    feature_data = scenario.feature_data.to_numpy()
    performance_data = scenario.performance_data.to_numpy()
    feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

    feature_data, performance_data, feature_cost_data = shuffle_in_unison(feature_data, performance_data, feature_cost_data)

    last_instance_id = amount_of_training_instances
    if amount_of_training_instances <= 0:
        last_instance_id = len(scenario.instances)

    start_time = time.time()
    for instance_id in range(0, last_instance_id):

        if instance_id % 100 == 0 and instance_id > 0:
            end_time = time.time()
            logger.info("Starting with instance " + str(instance_id)+ ". Last 100 instances took " + "{:.2f}".format(end_time-start_time) + " s .")
            start_time = time.time()

        X = feature_data[instance_id]
        y = performance_data[instance_id]

        # check if instance contains a non-censored value. If not, we will ignore it, as it does not have a ground truth
        contains_non_censored_value = False
        for y_element in y:
            if y_element < scenario.algorithm_cutoff_time:
                contains_non_censored_value = True
        if contains_non_censored_value:

            # compute feature time
            accumulated_feature_time = 0
            if scenario.feature_cost_data is not None and approach.get_name() != 'sbs' and approach.get_name() != 'oracle' and approach.get_name() != 'feature_free_epsilon_greedy':
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            #query prediction from learner
            predicted_scores = approach.predict(X, instance_id)
            predicted_algorithm_id = np.argmin(predicted_scores)

            #train learner with new sample
            approach.train_with_single_instance(X, predicted_algorithm_id, y[predicted_algorithm_id])

            #compute the values of the different metrics
            num_counted_test_values += 1
            for i, metric in enumerate(metrics):
                metric_result = metric.evaluate(y, predicted_scores, accumulated_feature_time, scenario.algorithm_cutoff_time)
                approach_metric_values[i] = (approach_metric_values[i] + metric_result)

    approach_metric_values = np.true_divide(approach_metric_values, num_counted_test_values)

    for i, metric in enumerate(metrics):
        print(metrics[i].get_name() + ': {0:.10f}'.format(approach_metric_values[i]))

    return approach_metric_values


def write_instance_wise_results_to_file(instancewise_result_strings: list, scenario_name: str):
    if not os.path.exists('output'):
        os.makedirs('output')
    complete_instancewise_result_string = '\n'.join(instancewise_result_strings)
    f = open("output/" + scenario_name + ".arff", "a")
    f.write(complete_instancewise_result_string + "\n")
    f.close()

def shuffle_in_unison(a:ndarray , b:ndarray, c:ndarray):
    assert len(a) == len(b)
    assert len(b) == len(c)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c