import logging
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import configparser
import multiprocessing as mp
import numpy as np
import database_utils
from evaluation import evaluate_scenario
from approaches.offline.baselines.per_algorithm_regressor import PerAlgorithmRegressor
from approaches.offline.baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from approaches.offline.baselines.sunny import SUNNY
from approaches.offline.baselines.snnap import SNNAP
from approaches.offline.baselines.isac import ISAC
from approaches.offline.baselines.satzilla11 import SATzilla11
from approaches.offline.baselines.satzilla07 import SATzilla07
from approaches.online.deegrote import Degroote
from approaches.online.cox_regression import CoxRegression
from approaches.online.feature_free_epsilon_greedy import FeatureFreeEpsilonGreedy
from approaches.online.bandit_selection_strategies.ucb import UCB
from approaches.online.bandit_selection_strategies.epsilon_greedy import EpsilonGreedy
from approaches.online.linUCB import LinUCBPerformance
from approaches.online.online_oracle import OnlineOracle
from approaches.online.thompson import Thompson
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from par_10_metric import Par10Metric
from par_10_regret_metric import Par10RegretMetric
from simple_runtime_metric import RuntimeMetric
from learner_runtime_metric import LearnerRuntimeMetric
from number_unsolved_instances import NumberUnsolvedInstances


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiments for scenario: " + result)


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        if approach_name == 'thompson_sensivity_sigma':
            for sigma in np.arange(1,10.5, 0.5):
                approaches.append(Thompson(sigma=sigma, lamda=0.5, buckley_james=True, revisited=True, true_expected_value=True))
        if approach_name == 'thompson_sensivity_lambda':
            for lamda in np.arange(0.05,1.05,0.05):
                approaches.append(Thompson(sigma=1.0, lamda=lamda, buckley_james=True, revisited=True, true_expected_value=True))
        if approach_name == 'linucb_sensivity_sigma':
            for sigma in np.arange(1,10.5, 0.5):
                approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1.0, sigma=sigma, new_tricks=True, revisited=True, ignore_censored=False, true_expected_value=True))
        if approach_name == 'linucb_sensivity_alpha':
            for alpha in np.arange(0.1,2.1,0.1):
                approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=alpha, sigma=10.0, new_tricks=True, revisited=True, ignore_censored=False, true_expected_value=True))
        if approach_name == 'linucb_sensivity_randsigma':
            for randsigma in np.arange(0.025,0.55,0.025):
                approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1.0, sigma=10.0, randsigma=randsigma, new_tricks=True, revisited=True, ignore_censored=False, true_expected_value=True))
        if approach_name == 'online_oracle':
            approaches.append(OnlineOracle())
        if approach_name == 'thompson':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=False, revisited=False, true_expected_value=False))
        if approach_name == 'thompson_rev':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=False, revisited=True, true_expected_value=False))
        if approach_name == 'bj_thompson':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=True, revisited=False, true_expected_value=False))
        if approach_name == 'bj_thompson_rev':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=True, revisited=True, true_expected_value=False))
        if approach_name == 'e_thompson':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=False, revisited=False, true_expected_value=True))
        if approach_name == 'e_thompson_rev':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=False, revisited=True, true_expected_value=True))
        if approach_name == 'bj_e_thompson':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=True, revisited=False, true_expected_value=True))
        if approach_name == 'bj_e_thompson_rev':
            approaches.append(Thompson(sigma=1.0, lamda=0.5, buckley_james=True, revisited=True, true_expected_value=True))
        if approach_name == 'blinducb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=False, revisited=False, ignore_censored=True, true_expected_value=False))
        if approach_name == 'blinducb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=False, revisited=True, ignore_censored=True, true_expected_value=False))
        if approach_name == 'rand_blinducb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=True, revisited=False, ignore_censored=True, true_expected_value=False))
        if approach_name == 'rand_blinducb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=True, revisited=True, ignore_censored=True, true_expected_value=False))
        if approach_name == 'bclinucb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=False, revisited=False, ignore_censored=False, true_expected_value=False))
        if approach_name == 'bclinucb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=False, revisited=True, ignore_censored=False, true_expected_value=False))
        if approach_name == 'rand_bclinucb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=True, revisited=False, ignore_censored=False, true_expected_value=False))
        if approach_name == 'rand_bclinucb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, new_tricks=True, revisited=True, ignore_censored=False, true_expected_value=False))
        if approach_name == 'e_blinducb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=False, revisited=False, ignore_censored=True, true_expected_value=True))
        if approach_name == 'e_blinducb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=False, revisited=True, ignore_censored=True, true_expected_value=True))
        if approach_name == 'e_rand_blinducb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=True, revisited=False, ignore_censored=True, true_expected_value=True))
        if approach_name == 'e_rand_blinducb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=True, revisited=True, ignore_censored=True, true_expected_value=True))
        if approach_name == 'e_bclinucb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=False, revisited=False, ignore_censored=False, true_expected_value=True))
        if approach_name == 'e_bclinucb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=False, revisited=True, ignore_censored=False, true_expected_value=True))
        if approach_name == 'e_rand_bclinucb':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=True, revisited=False, ignore_censored=False, true_expected_value=True))
        if approach_name == 'e_rand_bclinucb_rev':
            approaches.append(LinUCBPerformance(bandit_selection_strategy=UCB(gamma=1), alpha=1, sigma=10, new_tricks=True, revisited=True, ignore_censored=False, true_expected_value=True))
        if approach_name == 'degroote_epsilon_greedy':
            approaches.append(Degroote(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05)))
        if approach_name == 'degroote_linear_epsilon_greedy':
            approaches.append(Degroote(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05), regression_model=LinearRegression(n_jobs=1)))
        if approach_name == 'degroote_ucb':
            approaches.append(Degroote(bandit_selection_strategy=UCB(gamma=1)))
        if approach_name == 'cox_regression_epsilon_greedy':
            approaches.append(CoxRegression(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05), learning_rate=0.001))
        if approach_name == 'feature_free_epsilon_greedy_cutoff':
            approaches.append(FeatureFreeEpsilonGreedy(imputation_strategy='cutoff'))
        if approach_name == 'feature_free_epsilon_greedy_cutoff_sensitivity':
            for epsilon in np.arange(0.0,0.2, 0.01):
                approaches.append(FeatureFreeEpsilonGreedy(imputation_strategy='cutoff',bandit_selection_strategy=EpsilonGreedy(epsilon=epsilon)))
        if approach_name == 'feature_free_epsilon_greedy_par10':
            approaches.append(FeatureFreeEpsilonGreedy(imputation_strategy='par10'))
        if approach_name == 'per_algorithm_regressor':
            approaches.append(PerAlgorithmRegressor())
        if approach_name == 'imputed_per_algorithm_rf_regressor':
            approaches.append(PerAlgorithmRegressor(impute_censored=True))
        if approach_name == 'imputed_per_algorithm_ridge_regressor':
            approaches.append(PerAlgorithmRegressor(
                scikit_regressor=Ridge(alpha=1.0), impute_censored=True))
        if approach_name == 'multiclass_algorithm_selector':
            approaches.append(MultiClassAlgorithmSelector())
        if approach_name == 'sunny':
            approaches.append(SUNNY())
        if approach_name == 'snnap':
            approaches.append(SNNAP())
        if approach_name == 'satzilla-11':
            approaches.append(SATzilla11())
        if approach_name == 'satzilla-07':
            approaches.append(SATzilla07())
        if approach_name == 'isac':
            approaches.append(ISAC())
    return approaches



#######################
#         MAIN        #
#######################

initialize_logging()
config = load_configuration()
logger.info("Running experiments with config:")
print_config(config)

#fold = int(sys.argv[1])
#logger.info("Running experiments for fold " + str(fold))

db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(
    config)
database_utils.create_table_if_not_exists(db_handle, table_name)

amount_of_cpus_to_use = int(config['EXPERIMENTS']['amount_of_cpus'])
pool = mp.Pool(amount_of_cpus_to_use)


scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
path_to_scenario_folder = config["EXPERIMENTS"]["data_folder"]
approach_names = config["EXPERIMENTS"]["approaches"].split(",")
amount_of_scenario_training_instances = int(
    config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
debug_mode = config["EXPERIMENTS"]["debug_mode"] == 'True'
# we do not make a train/test split in the online setting as we have no prior training dataset. Accordingly,
# we only have one fold
for fold in range(1,11):
    for scenario in scenarios:
        approaches = create_approach(approach_names)

        if len(approaches) < 1:
            logger.error("No approaches recognized!")
        for approach in approaches:
            metrics = list()
            metrics.append(Par10Metric())
            metrics.append(Par10RegretMetric())
            metrics.append(RuntimeMetric())
            metrics.append(LearnerRuntimeMetric())
            if approach.get_name() != 'oracle':
                metrics.append(NumberUnsolvedInstances(False))
                metrics.append(NumberUnsolvedInstances(True))
            logger.info("Submitted pool task for approach \"" +
                        str(approach.get_name()) + "\" on scenario: " + scenario)
            if debug_mode:
                evaluate_scenario(scenario, path_to_scenario_folder, approach, metrics,
                                  amount_of_scenario_training_instances, fold, config)
                print('Finished evaluation of fold')
            else:
                pool.apply_async(evaluate_scenario, args=(scenario, path_to_scenario_folder, approach, metrics,
                                                          amount_of_scenario_training_instances, fold, config), callback=log_result)

pool.close()
pool.join()

logger.info("Finished all experiments.")
