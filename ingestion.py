import pandas as pd
import os
import glob
import json


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f) 


# Function for data ingestion
def merge_multiple_dataframe():
    ingested_f = open(os.path.join(config['output_folder_path'], 'ingestedfiles.txt'), 'w')

    full_df = pd.DataFrame()
    for csv_f in glob.glob(pathname=os.path.join(config['input_folder_path'], '*.csv')):
        df = pd.read_csv(csv_f)
        full_df = pd.concat([full_df, df], axis=0, ignore_index=True)
        ingested_f.write(f'{csv_f}\n')

    ingested_f.close()

    full_df = full_df.drop_duplicates(ignore_index=True)
    full_df.to_csv(os.path.join(config['output_folder_path'], 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
