#Import
import pandas as pd
import numpy as np

#Data
import pickle

#Model
from pysr import PySRRegressor, TensorBoardLoggerSpec
from sklearn.model_selection import train_test_split

# Import data
with open('data/df_filtered_rearranged.pkl', 'rb') as f:
    df = pickle.load(f)

df['mix'] = df['compA'] + df['compB']
mixtures = df['mix'].unique()

train_mix, test_mix = train_test_split(mixtures, test_size=0.3, random_state=42)

df = df[df['P'] != 77000]
X_columns = ['mA', 'mB', 'KA', 'KB']
X = df[X_columns]
y = df['K']

train_df = df[df['mix'].isin(train_mix)]
test_df = df[df['mix'].isin(test_mix)]

X_train = train_df[X_columns]
y_train = train_df['K']
X_test = test_df[X_columns]
y_test = test_df['K']

# Create a logger that writes to "logs/run*":
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)

model = PySRRegressor(
    logger_spec=logger_spec,
    niterations=5000,
    dimensional_constraint_penalty=10**5,
    parsimony=0.000001,
    warmup_maxsize_by = 0.75,
    turbo=True,
    populations=48,
)

model.fit(
    X_train, 
    y_train,
    X_units=["1", "1", "W*m^-1*K^-1", "W*m^-1*K^-1"],
    y_units="W*m^-1*K^-1"
    )