import os
import json
import shutil


# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


# function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    for fname in ['trainedmodel.pkl', 'latestscore.txt', 'ingestedfiles.txt']:
        shutil.copy(os.path.join(dataset_csv_path, fname),
                    os.path.join(prod_deployment_path, fname))

if __name__ == "__main__":
    store_model_into_pickle()
        
        

