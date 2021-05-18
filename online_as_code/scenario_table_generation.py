from plot_generation import load_configuration
from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np

def compute_statistics():
    config = load_configuration()

    scenario_names = config["EXPERIMENTS"]["scenarios"].split(",")
    path_to_scenario_folder = config["EXPERIMENTS"]["data_folder"]
    scenario_to_properties_dict = dict()
    for scenario_name in scenario_names:
        scenario = ASlibScenario()
        scenario.read_scenario(path_to_scenario_folder + scenario_name)
        scenario_to_properties_dict[scenario_name] = dict()
        properties_dict = scenario_to_properties_dict[scenario_name]

        properties_dict["num_instances"] = len(scenario.instances)
        properties_dict["num_features"] = len(scenario.feature_data.columns)
        properties_dict["num_algorithms"] = len(scenario.algorithms)
        properties_dict["cutoff-time"] = scenario.algorithm_cutoff_time
        compute_percentage_timeouts_for_scenario(scenario)




def compute_percentage_timeouts_for_scenario(scenario: ASlibScenario):
    num_instances = len(scenario.instances)
    num_algorithms = len(scenario.algorithms)

    timeouts_per_algorithm = dict()
    total_amount_of_timeouts = 0
    timeouts_per_instance = np.arra

    for instance_id in range(num_instances):
        timeouts_per_instance[instance_id] = 0

    for algorithm_id in range(num_algorithms):
        timeouts_per_algorithm[algorithm_id] = 0
        performances_of_algorithm = scenario.performance_data.iloc[:, algorithm_id].to_numpy()
        for instance_id in range(num_instances):
            if performances_of_algorithm[instance_id] >= scenario.algorithm_cutoff_time:
                timeouts_per_algorithm[algorithm_id] += 1
                total_amount_of_timeouts += 1
                timeouts_per_instance[instance_id] += 1
        print("a: " + str(algorithm_id) + " = " + str(timeouts_per_algorithm[algorithm_id]/num_instances))


compute_statistics()