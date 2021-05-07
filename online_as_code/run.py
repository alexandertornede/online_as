import logging
import os
import configparser
import multiprocessing as mp
import database_utils
from evaluation import evaluate_scenario
from approaches.offline.single_best_solver import SingleBestSolver
from approaches.offline.single_best_solver_with_feature_costs import SingleBestSolverWithFeatureCosts
from approaches.offline.virtual_single_best_solver import VirtualSingleBestSolverWithFeatureCosts
from approaches.offline.oracle import Oracle
from approaches.offline.survival_forests.surrogate import SurrogateSurvivalForest
from approaches.offline.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
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
from approaches.online.online_linear_regression import OnlineLinearRegression
from approaches.online.bandit_selection_strategies.ucb import UCB
from approaches.online.bandit_selection_strategies.epsilon_greedy import EpsilonGreedy
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from par_10_metric import Par10Metric
from par_10_regret_metric import Par10RegretMetric
from simple_runtime_metric import RuntimeMetric
from number_unsolved_instances import NumberUnsolvedInstances


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


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
        if approach_name == 'online_linear_regression_epsilon_greedy':
            approaches.append(OnlineLinearRegression(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05)))
        if approach_name == 'online_linear_regression_ucb':
            approaches.append(OnlineLinearRegression(bandit_selection_strategy=UCB(gamma=1), reward_strategy='b_j_motivated'))
        if approach_name == 'degroote_epsilon_greedy':
            approaches.append(Degroote(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05)))
        if approach_name == 'degroote_linear_epsilon_greedy':
            approaches.append(Degroote(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05), regression_model=LinearRegression(n_jobs=1)))
        if approach_name == 'degroote_ucb':
            approaches.append(Degroote(bandit_selection_strategy=UCB(gamma=1)))
        if approach_name == 'cox_regression_epsilon_greedy':
            approaches.append(CoxRegression(bandit_selection_strategy=EpsilonGreedy(epsilon=0.05), learning_rate=0.001))
        if approach_name == 'feature_free_epsilon_greedy':
            approaches.append(FeatureFreeEpsilonGreedy())
        if approach_name == 'sbs':
            approaches.append(SingleBestSolver())
        if approach_name == 'sbs_with_feature_costs':
            approaches.append(SingleBestSolverWithFeatureCosts())
        if approach_name == 'virtual_sbs_with_feature_costs':
            approaches.append(VirtualSingleBestSolverWithFeatureCosts())
        if approach_name == 'oracle':
            approaches.append(Oracle())
        if approach_name == 'ExpectationSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Expectation'))
        if approach_name == 'PolynomialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Polynomial'))
        if approach_name == 'GridSearchSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='GridSearch'))
        if approach_name == 'ExponentialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Exponential'))
        if approach_name == 'SurrogateAutoSurvivalForest':
            approaches.append(SurrogateAutoSurvivalForest())
        if approach_name == 'PAR10SurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='PAR10'))
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
            if approach.get_name() != 'oracle':
                metrics.append(NumberUnsolvedInstances(False))
                metrics.append(NumberUnsolvedInstances(True))
            logger.info("Submitted pool task for approach \"" +
                        str(approach.get_name()) + "\" on scenario: " + scenario)
            # pool.apply_async(evaluate_scenario, args=(scenario, path_to_scenario_folder, approach, metrics,
            #                                           amount_of_scenario_training_instances, fold, config), callback=log_result)

            evaluate_scenario(scenario, path_to_scenario_folder, approach, metrics,
                             amount_of_scenario_training_instances, fold, config)
            print('Finished evaluation of fold')

pool.close()
pool.join()

logger.info("Finished all experiments.")
