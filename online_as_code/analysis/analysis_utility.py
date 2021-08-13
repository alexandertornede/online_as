import configparser
import re

def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config

def clean_algorithm_name(name:str):
    short_name = re.sub(r'_sigma=.+', '', name)
    short_name = re.sub(r'^e_', '', short_name)
    short_name = re.sub(r'_e_', '_', short_name)
    return short_name