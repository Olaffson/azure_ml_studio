import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import pickle

from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.train.automl import AutoMLConfig

# run = experiment.start_logging()
run = Run.get_context()


df = pd.read_csv('../data/data_utilisable.csv')

X = df.drop("prix", axis=1)
y = df['prix']

numerical_cols = make_column_selector(dtype_include=np.number)
categorical_cols = make_column_selector(dtype_exclude=np.number)

numerical_pipeline = make_pipeline(SimpleImputer(), StandardScaler())
categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

preprocessor = make_column_transformer((numerical_pipeline, numerical_cols), (categorical_pipeline, categorical_cols))

model = make_pipeline(preprocessor, RandomForestRegressor())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

model.predict(X_test)

score = model.score(X_test, y_test)

print(model.score(X_test, y_test))

print(score)

# Enregistrez les métriques et les résultats de votre script d'apprentissage automatique
run.log("Score", score)
run.complete()

