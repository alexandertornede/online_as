from plot_generation import load_configuration
from plot_generation import get_dataframe_for_sql_query
from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_superset_vs_linear_plot():
    config = load_configuration()

    scenario_statistics = compute_statistics()

    scenario_names = config["EXPERIMENTS"]["scenarios"].split(",")

    dataframe = get_dataframe_for_sql_query("SELECT * FROM (SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `bugfix_in_bound_adaptive_lambda` WHERE metric='par10' GROUP BY scenario_name, approach, metric UNION SELECT * FROM `server_results_standard_v2_aggregated`) as T WHERE approach != 'super_set_online_linear_regression_UCB' AND approach NOT LIKE 'degroote%%' ORDER BY scenario_name, avg_result")

    y_values = list()
    x_values = list()
    for scenario in scenario_names:
        scenario_results = dataframe.loc[dataframe['scenario_name'] == scenario] # & dataframe['approach'] == 'super_set_online_linear_regression_UCB'
        super_set_result = scenario_results.loc[scenario_results['approach'] == 'super_set_online_linear_regression_lambda=0.5_UCB']['avg_result'].values[0]
        lr_result = scenario_results.loc[scenario_results['approach'] == 'online_linear_regression_cutoff_scaled_UCB']['avg_result'].values[0]
        division = super_set_result / lr_result
        y_values.append(division)

        x_values.append(scenario_statistics[scenario]['average_timeout_percentage'])

    print()
    plt.scatter(x_values, y_values)
    plt.axhline(y=1, color='r', linestyle='-')
    plt.xlabel("%C")
    plt.ylabel("super_set - LinUCB relation")

    regr = linear_model.LinearRegression()
    regr.fit(X=np.asarray(x_values).reshape(-1,1), y=y_values)
    dummy_x = np.arange(0,1,0.01)
    y_pred = regr.predict(dummy_x.reshape(-1,1))

    plt.plot(dummy_x, y_pred, linestyle="dashed")

    plt.show()
    print()
    # ax.set_xlabel("Scenario")
    # ax.set_ylabel("PAR10")
    # ax.legend(['oracle', 'AS-oracle', 'SBS', 'SBAS'])
    # vals = ax.get_yticks()
    # for tick in vals:
    #     ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#6d6e6d', zorder=1)
    #
    # name = 'plots/bars_small.pdf' if small_scenarios else 'plots/bars_large.pdf'
    # plt.savefig(name, bbox_inches = "tight")
    # print(dataframe.to_latex(index=False, float_format="%.3f"))

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

    properties_dict["timeout_percentage_per_algorithm"] = timeouts_per_algorithm/num_instances
    properties_dict["average_timeout_percentage"] = np.mean(timeouts_per_algorithm)/num_instances
    properties_dict["median_timeout_percentage"] = np.median(timeouts_per_algorithm)/num_instances
    properties_dict["min_timeout_percentage"] = np.min(timeouts_per_algorithm)/num_instances
    properties_dict["max_timeout_percentage"] = np.max(timeouts_per_algorithm)/num_instances

    # print("sorted: " + str(np.sort(timeouts_per_algorithm)/num_instances))
    # print("avg: " + str(np.mean(timeouts_per_algorithm)/num_instances))
    # print("median: " + str(np.median(timeouts_per_algorithm)/num_instances))
    # print("min: " + str(np.min(timeouts_per_algorithm/num_instances)))
    # print("max: " + str(np.max(timeouts_per_algorithm/num_instances)))


#compute_statistics()
generate_superset_vs_linear_plot()