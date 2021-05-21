from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import configparser
import re

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
        return pd.read_sql(sql_query, con=db_connection)

    def get_database_credential_string(self):
        return "mysql+pymysql://" + self.db_username + ":" + self.db_password + "@" + self.db_host + "/" + self.db_database


def plot(tablename: str, parameter: str) -> None:
    """
    Generate one plot for each scenario from the specified table where the parameter matches.
    :param tablename: name of the sql-table
    :param parameter: name of the parameter
    """

    table = database.get_query("SELECT scenario_name, approach, metric, AVG(result) as avg_result, COUNT(result), STDDEV(result) FROM %s WHERE metric='par10' GROUP BY scenario_name, approach, metric" % tablename)

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
        ax.set_xlim(left=0)
        ax.set_xlim(right=1)

        plt.title(scenario)
        if parameter == 'lambda':
            plt.xlabel(r'$\lambda$')
        else:
            plt.xlabel(parameter)
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
        fig.savefig("plots/%s.pdf" % scenario, bbox_inches='tight')



database = SQLConnection(config_path='../conf/experiment_configuration.cfg')
plot(tablename='sensitivity_analysis', parameter='lambda')
