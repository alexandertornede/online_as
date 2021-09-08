import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from analysis_utility import clean_algorithm_name

plt.style.use('seaborn-whitegrid')


def generate_latex_code_for_figure_inclusion(directory_of_figures:str, subfigure_size:float, sub_directory_in_latex: str, caption: str, label_prefix:str):
    latex_code = "\\begin{figure}[htb]\n\t\\centering\n"
    for i,filename in enumerate(sorted(os.listdir(directory_of_figures))):
        if filename.endswith(".pdf"):
            latex_code += "\\begin{subfigure}{" + str(subfigure_size) + "\\textwidth}\n"
            latex_code += "\t \includegraphics[width=\linewidth]{img/" + sub_directory_in_latex + "/" +  filename + "}\n"
            latex_code += "\t \\caption{}\n"
            latex_code += "\t \\label{" + label_prefix + "_" + filename.lower().replace('.pdf','') + "}\n"
            latex_code += "\\end{subfigure}\n"
        if i> 0 and i % 14 == 0:
            latex_code += "\\caption{" + caption + "}\n"
            latex_code += "\\label{" + label_prefix + "}\n"
            latex_code += "\\end{figure}\n"
            latex_code += "\\begin{figure}[htb]\n\t\\ContinuedFloat\n\t\\centering\n"
    latex_code += "\\caption{(Cont.) " + caption + "}\n"
    latex_code += "\\label{" + label_prefix + "_%s}\n" % str(int(i/14))
    latex_code += "\\end{figure}"
    print(latex_code)



def plot(directory, output_directory, title, xlabel, ylabel, approach_names=None, cumulative=False):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):

            meta_data = filename.replace('.txt', '').split('+')

            if approach_names is None or meta_data[2] in approach_names:
                with open(os.path.join(directory, filename)) as f:
                    data = [float(d) for d in f.read().replace('\n', '').split(';')]

                    if cumulative:
                        data = [sum(data[:i]) for i in range(1, len(data) + 1)]

                    dfs.append(pd.DataFrame(
                        {'scenario': meta_data[0], 'seed': meta_data[1], 'approach': meta_data[2], 'result': data}))

    try:
        df = pd.concat(dfs)
    except ValueError:
        sys.exit("no approach recognized - please check correct spelling and enter only name before first '_'.")

    df['result'] = pd.to_numeric(df['result'])

    for scenario in df['scenario'].unique():
        scenario_df = df.loc[df['scenario'] == scenario]

        fig = plt.figure()
        ax = fig.subplots()

        for approach in sorted(scenario_df['approach'].unique()):
            approach_scenario_df = scenario_df.loc[scenario_df['approach'] == approach]
            result = approach_scenario_df.groupby(level=0).mean()
            std = approach_scenario_df.groupby(level=0).std()

            x = np.arange(0, len(result))
            approach_short_name = clean_algorithm_name(approach)
            ax.plot(x, result, label=approach_short_name)
            ax.fill_between(x, (result - std).to_numpy().flatten(), (result + std).to_numpy().flatten(), alpha=.3)

        plt.legend(loc=2)

        plt.title(scenario)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        #plt.show()

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plt.savefig(output_directory + f'/{scenario}.pdf')


print("runtime plots")
plot(directory='../server_output/runtimes', output_directory='../figures/runtime_plots', title='Plot Title', xlabel='Timestep / #Instances', ylabel='Prediction time in s', cumulative=False, approach_names=['bj_e_thompson_rev_sigma=1.0_lambda=0.5','e_rand_bclinucb_rev_sigma=10_alpha=1_randsigma=0.25','degroote_EpsilonGreedy_RandomForestRegressor','degroote_UCB_RandomForestRegressor'])
generate_latex_code_for_figure_inclusion(directory_of_figures='../figures/runtime_plots', subfigure_size=0.3, sub_directory_in_latex='runtime_plots', caption='Prediction time in seconds of a selection of approaches for the corresponding scenario.', label_prefix='fig:app_runtime')

print("cumulative regret plots")
plot(directory='../server_output/regret', output_directory='../figures/cumulative_regret_plots', title='Plot Title', xlabel='Timestep / #Instances', ylabel='Cumulative PAR10 regret wrt. oracle', cumulative=True, approach_names=['bj_e_thompson_rev_sigma=1.0_lambda=0.5','e_rand_bclinucb_rev_sigma=10_alpha=1_randsigma=0.25','degroote_EpsilonGreedy_RandomForestRegressor','degroote_UCB_RandomForestRegressor'])
generate_latex_code_for_figure_inclusion(directory_of_figures='../figures/cumulative_regret_plots', subfigure_size=0.3, sub_directory_in_latex='cumulative_regret_plots', caption='Cumulative PAR10 regret wrt. oracle.', label_prefix='fig:app_cumulative_regret')