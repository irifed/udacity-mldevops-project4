import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions
from training import process_data

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)


def score_model():
    """
    In order to generate plots, you need to call the model prediction function that you created diagnostics.py in
    Step 3. The function will use the test data from the directory specified in the test_data_path entry of your
    config.json starter file as input dateset. You can use this function to obtain a list of predicted values from
    your model.

    After you obtain predicted values and actual values for your data, you can use these to generate a confusion
    matrix plot. Your reporting.py script should save your confusion matrix plot to a file in your workspace called
    confusionmatrix.png. The confusionmatrix.png file can be saved in the directory specified in the
    output_model_path entry of your config.json file.

    :return:
    """
    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    X_test, y_test = process_data(df, label='exited')

    preds = model_predictions(X_test)

    cm = confusion_matrix(preds, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(config['output_model_path'], 'confusionmatrix.png'))


if __name__ == '__main__':
    score_model()
