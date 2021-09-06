# Code for paper: Machine Learning for Online Algorithm Selection under Censored Feedback

This repository holds the code for our paper "Machine Learning for Online Algorithm Selection under Censored Feedback" by Alexander Tornede, Viktor Bengs and Eyke Hüllermeier. Regarding questions please contact alexander.tornede@upb.de .

Please cite this work as
```
@inproceedings{tornede2021ml4oas,
  title={Machine Learning for Online Algorithm Selection under Censored Feedback},
  author={Tornede, Alexander and Bengs, Viktor and H{\"u}llermeier, Eyke},
  booktitle={???},
  pages={???},
  year={???},
  organization={???}
}
```

## Abstract
In online algorithm selection (OAS), instances of an algorithmic problem class are presented to an agent one after another, and the agent has to quickly select a presumably best algorithm from a fixed set of candidate algorithms. For decision problems such as satisfiability (SAT), quality typically refers to the algorithm's runtime. As the latter is known to exhibit a heavy-tail distribution, an algorithm is normally stopped when exceeding a predefined upper time limit. As a consequence, machine learning methods used to optimize an algorithm selection strategy in a data-driven manner need to deal with right-censored samples, a problem that has received little attention in the literature so far. 
In this work, we revisit multi-armed bandit algorithms for OAS and discuss their capability of dealing with the problem. Moreover, we adapt them towards runtime-oriented losses, allowing for partially censored data while keeping a space- and time-complexity independent of the time horizon. In an extensive experimental evaluation on an adapted version of the ASlib benchmark, we demonstrate that some of them can update their models several orders of magnitude faster than existing methods at the cost of an acceptable drop in performance.

## Execution Details (Getting the Code To Run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below.

### 1. Configuration
In order to reproduce the results by running our code, we assume that you have a MySQL server with version >=5.7.9 running.

As a next step, you have to create a configuration file entitled `experiment_configuration.cfg` in the `conf` folder on the top level of your IDE project next to the `github_ci_install.txt` file. This configuration file defines which experiments will be executed and should contain the following information:

```
[DATABASE]
host = my.sqlserver.com
username = username
password = password
database = databasename
table = server_results_all_variants
ssl = true

[EXPERIMENTS]
scenarios=ASP-POTASSCO,BNSL-2016,CPMP-2015,CSP-2010,CSP-MZN-2013,CSP-Minizinc-Time-2016,GRAPHS-2015,MAXSAT-PMS-2016,MAXSAT-WPMS-2016,MAXSAT12-PMS,MAXSAT15-PMS-INDU,MIP-2016,PROTEUS-2014,QBF-2011,QBF-2014,QBF-2016,SAT03-16_INDU,SAT11-HAND,SAT11-INDU,SAT11-RAND,SAT12-ALL,SAT12-HAND,SAT12-INDU,SAT12-RAND,SAT15-INDU,TSP-LION2015
data_folder=data/
approaches=e_thompson,e_thompson_rev,bj_e_thompson,bj_e_thompson_rev,e_blinducb,e_blinducb_rev,e_rand_blinducb,e_rand_blinducb_rev,e_bclinucb,e_bclinucb_rev,e_rand_bclinucb,e_rand_bclinucb_rev,online_oracle,degroote_epsilon_greedy,degroote_linear_epsilon_greedy,degroote_ucb
amount_of_training_scenario_instances=-1
amount_of_cpus=6
; values are: 'standard' (all), 'clip_censored', 'ignore_censored'
censored_value_imputation=standard
debug_mode=False
```

You have to adapt all entries below the `[DATABASE]` tag according to your database server setup. The entries have the following meaning:
* `host`: the address of your database server
* `username`: the username the code can use to access the database
* `password`: the password the code can use to access the database
* `database`: the name of the database where you imported the tables
* `table`: the name of the table, where results should be stored. This is created automatically by the code if it does not exist yet and should NOT be created manually. (DO NOT CHANGE THIS ENTRY)
* `ssl`: whether ssl should be used or not

Entries below the `[EXPERIMENTS]` define which experiments will be run. The configuration above will produce the main results presented in the paper, i.e. the results needed to produce Figures 1 and 8, Tables 2 and 3. You might want to adapt the `amount_of_cpus` entry such that the experiments can be parallelized onto the amount of cores you entered.

### 2. Packages and Dependencies
For running the code several dependencies have to be fulfilled. The easiest way of getting there is by using [Anaconda](https://anaconda.org/). For this purpose you find an Anaconda environment definition called `online_as.yml` in the `anaconda` folder at the top-level of this project. Assuming that you have Anaconda installed, you can create an according environment with all required packages via

```
conda env create -f online_as.yml
``` 

which will create an environment named `online_as`. After it has been successfully installed, you can use 
```
conda activate online_as
```
to activate the environment and run the code (see step 4).

### 3. ASLib Data
Obviously, the code requires access to the ASLib scenarios in order to run the requested evaluations. It expects the ASLib scenarios (which can be downloaded from [Github](https://github.com/coseal/aslib_data)) to be located in a folder `data` on the top-level of your IDE project. I.e. your folder structure should look similar to this: 
```
./online_as_code
./online_as_code/anaconda
./online_as_code/analysis
./online_as_code/approaches
./online_as_code/conf
./online_as_code/data
./online_as_code/figures
```

### 4. Evaluation Results
At this point you should be good to go and can execute the experiments by running the `run.py` on the top-level of the project. 

All results will be stored in the table given in the configuration file and has the following columns:

* `scenario_name`: The name of the scenario.
* `fold`: The train/test-fold associated with the scenario which is considered for this experiment
* `approach`: The approach which achieved the reported results.
* `metric`: The metric which was used to generate the result. For the `number_unsolved_instances` metric, the suffix `True` indicates that feature costs are accounted for whereas for `False` this is not the case. All other metrics automatically incorporate feature costs.
* `result`: The output of the corresponding metric.

### 5. Generating Tables and Figures
After you have successfully run the code and found the corresponding table filled in your database (cf. Sec. 4), you can generate Figure 1 and Tables 2 and 3 by running the `analysis/plot_generation.py` file. As a result, the latex code for the corresponding tables will be printed onto the console. Moreover, the plots included in Figure 1 will be stored in a `figures` directory at the top-level of your project. 

In order to generate Figure 8, rename the `output` folder, which was created during the run of the code, to `server_output`. It contains the necessary data to generate the plots included in Figure 8. After the renaming, you can run `analysis/plots_over_time.py`, which will print the Latex code for Figure 8 onto the console and store all required files in the `figures/runtime_plots` directory.

#### *Sensitivity Analysis*
In order to generate the sensitivity analysis, you have to run another set of experiments. For this purpose, replace the `approches` and `table` lines in the `experiment_configuration.cfg` file with the following two lines:

```
[DATABASE]
...
table = server_sensitivity
...

[EXPERIMENTS]
...
approaches=thompson_sensivity_sigma,thompson_sensivity_lambda,linucb_sensivity_sigma,linucb_sensivity_alpha,linucb_sensivity_randsigma
...
```

Now, run the `run.py` file again and wait until all experiment have been finished (as the console will tell you). Once this is done, you can run the `analysis/sensitivity_plotter.py` file in order to print the Latex code for Figures 3-7 onto the console and generate the corresponding plots in the directory `figures/sensitivity`. 

## Hyperparameter Details
All details regarding the hyperparameters and their settings can be found in the `appendix.pdf` file on the top level of the project.
