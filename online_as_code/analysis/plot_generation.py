import configparser
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config

def bold_extreme_values(data, best=-1, second_best=-1):

    if data == best:
        return "\\textbf{%s}" % "{:.3f}".format(data)

    if data == second_best:
        return "\\underline{%s}" % "{:.3f}".format(data)

    return "{:.3f}".format(data)

def generate_result_table(sql_query:str):
    dataframe = get_dataframe_for_sql_query(sql_query)
    df = dataframe.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')

    for k in range(len(df.index)):
        df.iloc[k] = df.iloc[k].apply(
            lambda data: bold_extreme_values(data, best=df.iloc[k].min(), second_best=np.partition(df.iloc[k].array.to_numpy(), 1)[1]))

    # Set column header to bold title case
    df.columns = (df.columns.to_series()
                   .apply(lambda r:
        r.replace("_", " ").title()))

    # Write to file
    # format = "l" + \
    #          "@{\hskip 12pt}" + \
    #          4*"S[table-format = 2.2]"

    print(df.to_latex(index=True, escape=False))

def get_dataframe_for_sql_query(sql_query: str ):
    db_credentials = get_database_credential_string()
    return pd.read_sql(sql_query, con=db_credentials)

def get_database_credential_string():
    config = load_configuration()
    db_config_section = config['DATABASE']
    db_host = db_config_section['host']
    db_username = db_config_section['username']
    db_password = db_config_section['password']
    db_database = db_config_section['database']
    return "mysql://" + db_username + ":" + db_password + "@" + db_host + "/" + db_database

def generate_preliminary_result_table():
    dataframe = get_dataframe_for_sql_query("SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `server_results_standard_all_v2` WHERE metric='par10' GROUP BY scenario_name, approach, metric ORDER BY scenario_name, avg_result")
    dataframe = dataframe.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')
    print(dataframe.to_latex(index=False, float_format="%.3f"))

#generate_level_N_normalized_par10_table(level=1)
#generate_normalized_par10_table_normalized_by_level_0()
#generate_sbs_vbs_change_plots(True)
#generate_sbs_vbs_change_plots(False)
#generate_preliminary_result_table()
print("FULL:")
generate_result_table("SELECT * FROM (SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `bugfix_in_bound_adaptive_lambda` WHERE metric='par10' GROUP BY scenario_name, approach, metric UNION SELECT * FROM `server_results_standard_v2_aggregated`) as T WHERE approach != 'super_set_online_linear_regression_UCB' ORDER BY scenario_name, avg_result")
print("PARTIAL:")
generate_result_table("SELECT * FROM (SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `bias_linucb` WHERE metric='par10' GROUP BY scenario_name, approach, metric UNION SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `bugfix_in_bound_adaptive_lambda` WHERE metric='par10' GROUP BY scenario_name, approach, metric UNION SELECT * FROM `server_results_standard_v2_aggregated` WHERE approach = 'feature_free_epsilon_greedy') as T WHERE approach != 'super_set_online_linear_regression_UCB' AND approach NOT LIKE 'degroote%%' ORDER BY scenario_name, avg_result")