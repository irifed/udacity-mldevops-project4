import numpy as np
import pandas as pd
import pickle
import os
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


def process_data(X, label=None):
    """ Process the data used in the machine learning pipeline.

    This can be used in either training or inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Preprocess data

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label.
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)

    Returns
    -------
    X : np.array
        Processed data for training or inference. Missing data is imputed if training.
    y : np.array
        Processed labels if training=True, otherwise empty np.array.
    """

    keep_cols = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees'
    ]

    if label is not None:
        y = X[label]
    else:
        y = np.array([])

    X = X[keep_cols]

    return X, y


# Function for training the model
def train_model():

    df = pd.read_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'))
    X_full, y_full = process_data(df, label='exited')

    # train/test split???
    X_train, y_train = X_full, y_full

    # preprocessor = ColumnTransformer(
    #     [('onehot', OneHotEncoder(dtype='int'), cat_features)],
    #     remainder='passthrough'
    # )

    clf = Pipeline(steps=[
        # ('preprocessor', preprocessor),
        ('model', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False))
    ])

    clf.fit(X_train, y_train)

    acc = clf.score(X_train, y_train)
    print(f'Fit accuracy = {acc}')

    joblib.dump(clf, os.path.join(config['output_folder_path'], 'trainedmodel.pkl'))


if __name__ == "__main__":
    train_model()