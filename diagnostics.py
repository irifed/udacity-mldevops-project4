import joblib
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import sys

from training import process_data

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 


# Function to get model predictions
def model_predictions(X_test):
    clf = joblib.load(os.path.join(config['output_model_path'], 'trainedmodel.pkl'))
    preds = clf.predict(X_test)

    return preds


# Function to get summary statistics
def dataframe_summary():
    """
    The summary statistics you should calculate are means, medians, and standard deviations.
    You should calculate each of these for each numeric column in your data.

    This function should calculate these summary statistics for the dataset stored in the directory
    specified by output_folder_path in config.json. It should output a Python list containing all of the summary
    statistics for every numeric column of the input dataset.

    :return:
    """
    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))

    stats = df.describe(include=np.number).loc[['mean', '50%', 'std']]
    return stats


def missing_data():
    """
    Next, you should write a function to check for missing data. By missing data, we mean NA values. Remember that
    the Pandas module has a custom method for checking whether a value is NA.

    Your function needs to count the number of NA values in each column of your dataset. Then, it needs to calculate
    what percent of each column consists of NA values.

    The function should count missing data for the dataset stored in the directory specified by output_folder_path in
    config.json. It will return a list with the same number of elements as the number of columns in your dataset.
    Each element of the list will be the percent of NA values in a particular column of your data.

    :return:
    """
    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))

    #na_stats = list(df.isna().sum() / df.shape[1])

    na_stats = df.isna().sum() / df.shape[1]
    return na_stats


def execution_time():
    """
    calculate timing of training.py and ingestion.py
    :return:
    """

    t_start = timeit.default_timer()
    os.system('python3 ingestion.py')
    t_ingestion = timeit.default_timer() - t_start

    t_start = timeit.default_timer()
    os.system('python3 training.py')
    t_training = timeit.default_timer() - t_start

    return [t_ingestion, t_training]


def check_versions(name):
    """
    Code for checking latest and current versions is borrowed from this StackOverflow thread:
    https://stackoverflow.com/questions/58648739/how-to-check-if-python-package-is-latest-version-programmatically

    :param name:
    :return:
    """
    latest_version = str(
        subprocess.run([sys.executable, '-m', 'pip', 'install', '{}==random'.format(name)], capture_output=True,
                       text=True))
    latest_version = latest_version[latest_version.find('(from versions:') + 15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ', '').split(',')[-1]

    current_version = str(
        subprocess.run([sys.executable, '-m', 'pip', 'show', '{}'.format(name)], capture_output=True, text=True))
    version_pos = current_version.find('Version:')
    if version_pos > 0:
        current_version = current_version[version_pos + 8:]
        current_version = current_version[:current_version.find('\\n')].replace(' ', '')
    else:
        current_version = 'not_installed'

    return [current_version, latest_version]


def outdated_packages_list():
    with open('requirements.txt', 'rt') as fin, open('package_versions.txt', 'wt') as fout:
        for line in fin:
            package_name = line[:line.find('==')]
            cv, lv = check_versions(package_name)
            fout.write(f'{package_name}\t\t{cv}\t{lv}\n')


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    X_test, y_test = process_data(df, label='exited')

    model_predictions(X_test)
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
