import subprocess
import os
import json

with open('config.json','r') as f:
    config = json.load(f)

#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000/"

def run_apicalls():
    response1 = subprocess.run(['curl', '-X', 'POST', f'{URL}/prediction?filename=testdata/testdata.csv'],
                               capture_output=True).stdout.decode()

    response2 = subprocess.run(['curl', f'{URL}/scoring'], capture_output=True).stdout.decode()

    # problem: summarystats returns html-formatted dataframe, but in the apireturns.txt I need just plain text...
    response3 = subprocess.run(['curl', f'{URL}/summarystats'], capture_output=True).stdout.decode()

    response4 = subprocess.run(['curl', f'{URL}/diagnostics'], capture_output=True).stdout.decode()

    #combine all API responses
    responses = response1 + response2 + response3 + response4

    #write the responses to your workspace
    with open(os.path.join(config['output_model_path'], 'apireturns.txt'), 'wt') as f:
        f.write(responses)

if __name__ == "__main__":
    run_apicalls()