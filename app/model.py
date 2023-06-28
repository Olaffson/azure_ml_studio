import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.train.automl import AutoMLConfig

# Créez un objet Workspace en utilisant vos informations d'identification Azure
ws = Workspace.get(name="olive_ml",
                   subscription_id='111aaa69-41b9-4dfd-b6af-2ada039dd1ae',
                   resource_group='okotwica.ext-rg')

compute_target = ws.compute_targets['CumputeDS1V2']

myenv = Environment.from_pip_requirements(name='myenv', file_path='../requirements.txt')

src = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    compute_target=compute_target,
    environment=myenv)

# Créez un objet Experiment pour exécuter votre script d'apprentissage automatique
experiment = Experiment(workspace=ws, name="test_vscode_azure_ok")

# run = experiment.start_logging()
run = experiment.submit(config=src)

run.wait_for_completion(show_output=True)


# en tapant 'src'
# from azureml.core import Workspace, Experiment, ScriptRunConfig

# # get workspace
# ws = Workspace.from_config()

# # get compute target
# target = ws.compute_targets['target-name']

# # get registered environment
# env = ws.environments['env-name']

# # get/create experiment
# exp = Experiment(ws, 'experiment_name')

# # set up script run configuration
# config = ScriptRunConfig(
#     source_directory='.',
#     script='script.py',
#     compute_target=target,
#     environment=env,
#     arguments=['--meaning', 42],
# )

# # submit script to AML
# run = exp.submit(config)
# print(run.get_portal_url()) # link to ml.azure.com
# run.wait_for_completion(show_output=True)

