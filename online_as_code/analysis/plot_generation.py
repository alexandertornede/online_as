import configparser
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sqlalchemy import create_engine, text
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config

def bold_extreme_values(data, best=-1, second_best=-1, decimal_places:int  = 3):
    format_string = ("{:." + str(decimal_places) + "f}")
    if data == best:
        return "\\textbf{%s}" % format_string.format(data)

    if data == second_best:
        return "\\underline{%s}" % format_string.format(data)

    return format_string.format(data)

def generate_result_table(sql_query:str):
    dataframe = get_dataframe_for_sql_query(sql_query)
    df = dataframe.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')
    df_rank= df.rank(axis=1)
    df.loc['avgrank'] = df_rank.mean()

    for k in range(len(df.index)):
        df.iloc[k] = df.iloc[k].apply(
            lambda data: bold_extreme_values(data, best=df.iloc[k].min(), second_best=np.partition(df.iloc[k].array.to_numpy(), 1)[1]))

    # Set column header to bold title case
    df.columns = (df.columns.to_series()
                  .apply(lambda r:
                         '\\begin{rotate}{90}' + r.replace("_", " ").title() + '\\end{rotate}'))

    print(df.to_latex(index=True, escape=False))

def generate_result_table_with_ranks(sql_query:str):
    dataframe = get_dataframe_for_sql_query(sql_query)
    df = dataframe.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')
    df= df.rank(axis=1)
    df.loc['mean'] = df.mean()

    for k in range(len(df.index)):
        df.iloc[k] = df.iloc[k].apply(
            lambda data: bold_extreme_values(data, best=df.iloc[k].min(), second_best=np.partition(df.iloc[k].array.to_numpy(), 1)[1], decimal_places=2))

    # Set column header to bold title case
    df.columns = (df.columns.to_series()
                  .apply(lambda r:
                         '\\begin{rotate}{90}' + r.replace("_", " ").title() + '\\end{rotate}'))

    print(df.to_latex(index=True, escape=False))

def generate_npar10_dataframe(sql_query: str):
    online_oracle_df = get_dataframe_for_sql_query("SELECT scenario_name, approach, AVG(result) as avg_result FROM `server_results_all_variants` WHERE metric='par10' and approach = 'online_oracle' GROUP BY scenario_name, approach, metric ORDER BY scenario_name")
    online_oracle_df = online_oracle_df.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')
    online_oracle_series = pd.Series(online_oracle_df['online_oracle'], online_oracle_df.index)
    ff_greedy_df = get_dataframe_for_sql_query("SELECT scenario_name, approach, AVG(result) as avg_result FROM `server_results_all_variants` WHERE metric='par10' and approach = 'feature_free_epsilon_greedy_cutoff' GROUP BY scenario_name, approach, metric ORDER BY scenario_name")
    ff_greedy_df = ff_greedy_df.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')
    ff_greedy_series = pd.Series(ff_greedy_df['feature_free_epsilon_greedy_cutoff'], ff_greedy_df.index)

    dataframe = get_dataframe_for_sql_query(sql_query)
    df = dataframe.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')


    denominator = ff_greedy_df.sub(online_oracle_series, axis='index')
    denominator_series = pd.Series(denominator['feature_free_epsilon_greedy_cutoff'], denominator.index)

    nominator = df.sub(online_oracle_series, axis='index')

    npar10_df = (nominator).div(denominator_series, axis='index')
    return npar10_df

def generate_ablation_plots(algorithm: str):
    if algorithm == 'ucb':
        algorithm_stub = 'lin'
    if algorithm == 'thompson':
        algorithm_stub = 'thom'
    time_query = "SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `server_results_all_variants` WHERE metric='learner_runtime_s_per_step' AND approach LIKE '%e\_%' AND  approach LIKE '%" + algorithm_stub +"%' AND approach != 'online_oracle' GROUP BY scenario_name, approach, metric ORDER BY scenario_name, avg_result"
    time_df = get_dataframe_for_sql_query(time_query)
    time_df = time_df.pivot_table(values='avg_result', index='scenario_name', columns='approach', aggfunc='first')
    time_mean = time_df.mean()
    par10_query = "SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `server_results_all_variants` WHERE metric='par10' AND approach LIKE '%e\_%' AND  approach LIKE '%" + algorithm_stub +"%' AND approach != 'online_oracle' GROUP BY scenario_name, approach, metric ORDER BY scenario_name, avg_result"
    npar10_df = generate_npar10_dataframe(par10_query)
    npar10_mean = npar10_df.mean()

    fig, ax = plt.subplots()
    markers=['o', 'x', 's', 'd', '+', '^','p', '*']
    for i,name in enumerate(npar10_mean.index):
        short_name = re.sub(r'_sigma=.+', '', name.replace('e_', ''))
        ax.scatter(x=time_mean.loc[name], y=npar10_mean.loc[name], label=short_name, marker=markers[i])
    ax.legend()
    plt.ylabel('npar10')
    plt.xlabel('prediction time in s')

    figure_directory = 'figures'
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    plt.savefig(figure_directory + '/ablation_' + algorithm + '.pdf')


def get_dataframe_for_sql_query(sql_query: str ):
    db_credentials = get_database_credential_string()
    return pd.read_sql(text(sql_query), con=db_credentials)

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
#generate_result_table_with_ranks("SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `server_results_all_variants` WHERE approach LIKE '%e\_%' AND metric='par10' AND approach not LIKE 'e_thompson%' AND approach not LIKE 'bj_e_thompson_sig%' AND approach != 'feature_free_epsilon_greedy_par10' GROUP BY scenario_name, approach, metric ORDER BY scenario_name, avg_result")
#generate_result_table("SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result) FROM `server_results_all_variants` WHERE approach LIKE '%e\_%' AND metric='par10' AND approach not LIKE 'e_thompson%' AND approach not LIKE 'bj_e_thompson_sig%' AND approach != 'feature_free_epsilon_greedy_par10' GROUP BY scenario_name, approach, metric ORDER BY scenario_name, avg_result")
generate_ablation_plots('ucb')
generate_ablation_plots('thompson')