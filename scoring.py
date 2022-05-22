import pandas as pd
import os
import json
import joblib

from sklearn.metrics import f1_score

from training import process_data

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


# Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    X_test, y_test = process_data(df, label='exited')

    clf = joblib.load(os.path.join(config['output_folder_path'], 'trainedmodel.pkl'))

    preds = clf.predict(X_test)
    f1 = f1_score(preds, y_test)

    with open(os.path.join(config['output_folder_path'], 'latestscore.txt'), 'w') as f:
        f.write(f'{f1:.04f}\n')


if __name__ == "__main__":
    score_model()
