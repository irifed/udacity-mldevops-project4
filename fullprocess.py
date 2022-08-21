import os
import glob
import json
import joblib

import ingestion
import training
import scoring
import deployment
import reporting
import apicalls

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)


def is_file_new(fname):
    """
    Check if fname is in the ingestedfiles.txt.

    :param fname:
    :return:
        True if fname is not in the ingestedfiles.txt
    """
    ingested_files_path = os.path.join(config['prod_deployment_path'], 'ingestedfiles.txt')
    with open(ingested_files_path, 'rt') as ingested_files:
        for line in ingested_files:
            if fname == line.strip():
                return False

    return True


def new_input_found():
    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    input_folder = config['input_folder_path']
    print(f'checking if folder {input_folder} contains new files as compared to ingestedfiles.txt')
    for fname in glob.glob(pathname=os.path.join(input_folder, '*.csv')):
        print(f'checking file {fname}')
        if is_file_new(fname):
            return True
    return False


def get_score():
    score_fname = os.path.join(config['prod_deployment_path'], 'latestscore.txt')
    with open(score_fname, 'rt') as fscore:
        score = float(fscore.readline())
    return score


if __name__ == '__main__':
    if not new_input_found():
        print('new input not found, stopping.')
        exit(0)

    print('new input found, starting ingestion...')
    ingestion.merge_multiple_dataframe()
    # new data now is in os.path.join(config['output_folder_path'], 'finaldata.csv')

    old_model = joblib.load(os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl'))
    old_score = get_score()

    new_score = scoring.score_model(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    print(f'old_score = {old_score}, new_score = {new_score}')
    if abs(old_score - new_score) < 0.01:
        print('model score is the same, stopping')
        exit(0)

    force_retrain = True
    if not force_retrain and new_score > old_score:
        print('model score had improved, nothing to do, stopping')
        exit(0)

    print('model score had degraded, retraining...')
    # train, save new model, check score, compare with original score

    training.train_model()
    retrained_score = scoring.score_model(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    print(f'old_score = {old_score}, new_score = {new_score}, retrained_score = {retrained_score}')

    print('deploying retrained model...')
    deployment.store_model_into_pickle()

    print('saving new confusion matrix')
    reporting.score_model()

    print('saving new api calls output')
    apicalls.run_apicalls()

