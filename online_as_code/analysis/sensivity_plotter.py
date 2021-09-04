from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import configparser
import re
from analysis_utility import create_directory_if_not_exists, clean_algorithm_name
from sqlalchemy import create_engine, text

class SQLConnection:

    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read_file(open(config_path))
        self.db_username = config["DATABASE"]["username"]

        self.db_password = config["DATABASE"]["password"]
        self.db_host = config["DATABASE"]["host"]
        self.db_database = config["DATABASE"]["database"]

    def get_query(self, sql_query):
        return self.get_dataframe_for_sql_query(sql_query)

    def get_dataframe_for_sql_query(self, sql_query: str):
        db_credentials = self.get_database_credential_string()
        db_connection = create_engine(db_credentials)
        return pd.read_sql(text(sql_query), con=db_connection)

    def get_database_credential_string(self):
        return "mysql+pymysql://" + self.db_username + ":" + self.db_password + "@" + self.db_host + "/" + self.db_database


def plot(tablename: str, parameter: str, output_directory:str, approach_like:str, x_min:float, x_max:float) -> None:
    """
    Generate one plot for each scenario from the specified table where the parameter matches.
    :param tablename: name of the sql-table
    :param parameter: name of the parameter
    """
    sql_query = "SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result), STDDEV(result) FROM %s WHERE metric='par10' AND approach LIKE '%s' GROUP BY scenario_name, approach, metric" % (tablename, approach_like)
    table = database.get_query(sql_query)

    table['approach'] = table['approach'].str.replace('_UCB', '0_UCB')

    scenario_tables = [
        table.loc[(table['approach'].str.contains(parameter)) & (table['scenario_name'] == scenario)].sort_values(
            by=['approach']) for scenario in table.scenario_name.unique()]

    for scenario, data in zip(table.scenario_name.unique(), scenario_tables):
        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot()
        ax.grid(axis='y', linestyle='-', linewidth=1)
        # ax.set_ylim(bottom=800)
        # ax.set_ylim(top=1000)
        ax.set_xlim(left=x_min, right=x_max)

        plt.title(scenario)
        plt.xlabel(r'$\%s$' % parameter.replace("randsigma","widetilde{\sigma}^2"))
        plt.ylabel("PAR10")

        a = data['approach'].array.to_numpy()

        x = list()
        for approach in a:
            regex = r'%s=\d+\.\d+' %parameter
            relevant_name_part = re.findall(regex,approach)[0]
            relevant_name_part = relevant_name_part.replace(parameter+ "=", "")
            x.append(float(relevant_name_part))

        y = data['avg_result'].array.to_numpy()
        yerr = data['STDDEV(result)'].array.to_numpy()

        ax.errorbar(x, y, yerr=yerr, linestyle='none', marker='o')
        plt.show()

        create_directory_if_not_exists(output_directory)
        cleaned_approach_name = clean_algorithm_name(approach_like)
        filename = '/%s_%s_%s.pdf' % (scenario, parameter, cleaned_approach_name)
        fig.savefig(output_directory + filename, bbox_inches='tight')

    print_latex_figure_code(scenarios=table.scenario_name.unique(), approach=cleaned_approach_name, parameter=parameter)


def print_latex_figure_code(approach, parameter, scenarios):
    code = '\\begin{figure}[htb]\n\t\\centering\n'
    for scenario in sorted(scenarios):
        code += '\\begin{subfigure}{0.25\\textwidth}\n'
        code += '\t\\includegraphics[width=\linewidth]{img/sensitivity/%s_%s_%s.pdf}\n' % (scenario, parameter, approach)
        code += '\t\\label{fig:app_sensitivity_%s_%s_%s}\n' % (scenario, parameter, approach)
        code += '\\end{subfigure}\n'
    code += '\\caption{Sensitivity analysis for parameter %s of approach %s.}\n' % ('$\\' + parameter.replace("randsigma","widetilde{\sigma}^2") + '$', approach.replace('_','\_'))
    code += '\\label{fig:app_sensitivity_%s_%s}\n' % (approach, parameter)
    code += '\\end{figure}\n'
    print(code)



database = SQLConnection(config_path='../conf/experiment_configuration.cfg')
#sigma analysis of Thompson
plot(tablename='server_sensitivity', parameter='sigma', output_directory='../figures/sensitivity', approach_like='bj_e_thompson_rev_sigma=%_lambda=0.5', x_min=0.5, x_max=10.5)
#lambda analysis of Thompson
plot(tablename='server_sensitivity', parameter='lambda', output_directory='../figures/sensitivity', approach_like='bj_e_thompson_rev_sigma=1.0_lambda=%', x_min=0, x_max=1.1)
#sigma analysis of LinUCB
plot(tablename='server_sensitivity', parameter='sigma', output_directory='../figures/sensitivity', approach_like='e_rand_bclinucb_rev_sigma=%_alpha=1.0_randsigma=0.25', x_min=0.5, x_max=10.5)
#alpha analysis of LinUCB
plot(tablename='server_sensitivity', parameter='alpha', output_directory='../figures/sensitivity', approach_like='e_rand_bclinucb_rev_sigma=10.0_alpha=%_randsigma=0.25', x_min=0, x_max=2.1)
#randsimga analysis of LinUCB
plot(tablename='server_sensitivity', parameter='randsigma', output_directory='../figures/sensitivity', approach_like='e_rand_bclinucb_rev_sigma=10.0_alpha=1.0_randsigma=%', x_min=0, x_max=0.6)