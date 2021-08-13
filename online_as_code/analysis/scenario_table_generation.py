from analysis_utility import load_configuration
from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
from pandas import DataFrame

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
        compute_percentage_timeouts_for_scenario(scenario, properties_dict)
        scenario_to_properties_dict[scenario_name] = properties_dict
    return scenario_to_properties_dict



def compute_percentage_timeouts_for_scenario(scenario: ASlibScenario, properties_dict: dict):
    print(scenario.scenario)
    num_instances = len(scenario.instances)
    num_algorithms = len(scenario.algorithms)

    timeouts_per_algorithm = np.zeros(num_algorithms)
    total_amount_of_timeouts = 0
    timeouts_per_instance = np.zeros(num_instances)

    for algorithm_id in range(num_algorithms):
        performances_of_algorithm = scenario.performance_data.iloc[:, algorithm_id].to_numpy()
        for instance_id in range(num_instances):
            if performances_of_algorithm[instance_id] >= scenario.algorithm_cutoff_time:
                timeouts_per_algorithm[algorithm_id] += 1
                total_amount_of_timeouts += 1
                timeouts_per_instance[instance_id] += 1
        # print("a: " + str(algorithm_id) + " = " + str(timeouts_per_algorithm[algorithm_id]/num_instances))

    #properties_dict["timeout_percentage_per_algorithm"] = timeouts_per_algorithm/num_instances
    properties_dict["average_timeout_percentage"] = np.mean(timeouts_per_algorithm)/num_instances
    properties_dict["median_timeout_percentage"] = np.median(timeouts_per_algorithm)/num_instances
    properties_dict["min_timeout_percentage"] = np.min(timeouts_per_algorithm)/num_instances
    properties_dict["max_timeout_percentage"] = np.max(timeouts_per_algorithm)/num_instances

    # print("sorted: " + str(np.sort(timeouts_per_algorithm)/num_instances))
    # print("avg: " + str(np.mean(timeouts_per_algorithm)/num_instances))
    # print("median: " + str(np.median(timeouts_per_algorithm)/num_instances))
    # print("min: " + str(np.min(timeouts_per_algorithm/num_instances)))
    # print("max: " + str(np.max(timeouts_per_algorithm/num_instances)))


scenario_to_properties_dict = compute_statistics()
dataframe = DataFrame.from_dict(scenario_to_properties_dict, orient='index', columns=['num_instances','num_features','num_algorithms','cutoff-time','average_timeout_percentage','median_timeout_percentage','min_timeout_percentage','max_timeout_percentage'])
print(dataframe.to_latex(float_format="%.2f", header=['#I','#F','#A','C','T','MT','minT','maxT']))