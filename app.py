from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import joblib

import json
import os

from training import process_data
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

#train_data_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

prediction_model = joblib.load(os.path.join(config['output_model_path'], 'trainedmodel.pkl'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    endpoint should take a dataset's file location as its input, and return the outputs of the prediction function
    you created in Step 3.
    :return:
    """
    filename = request.args.get('filename')
    df = pd.read_csv(os.path.join(filename))
    X_test, y_test = process_data(df, label='exited')

    preds = model_predictions(X_test)
    return str(preds)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    f1 = score_model(test_data_path)
    return f'{f1:.2f}'

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    summary = dataframe_summary().to_html()
    return summary

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    This endpoint needs to run the timing, missing data, and dependency check functions you created in Step 3 and
    return their outputs.

    :return:
    """

    t_ingestion, t_training = execution_time()
    na_stats = missing_data()

    # outdated_packages_list()
    # show contents of 'package_versions.txt'

    output = f'<pre>Ingestion time: {t_ingestion:.2f}</pre><br>' \
             f'<pre>Training time: {t_training:.2f}<br>' \
             f'<pre>Missing values: {na_stats}</pre><br>'

    output2 = '<pre>'
    with open('package_versions.txt', 'rt') as f:
        for line in f:
            output2 += f'{line}'
    output2 += '</pre>'

    return output + output2


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
