#Import
import pandas as pd
import numpy as np

#Data
import pickle

#Model
from pysr import PySRRegressor, TensorBoardLoggerSpec
from sklearn.model_selection import train_test_split

# Import data
with open('../data/df_filtered_reduced.pkl', 'rb') as f:
    df = pickle.load(f)

# Rearrange condition

df['mix'] = df['compA'] + df['compB']
mixtures = df['mix'].unique()

train_mix, test_mix = train_test_split(mixtures, test_size=0.3, random_state=42)

df = df[df['P'] != 77000]
X_columns = ['mA', 'mB', 'KA', 'KB']
X = df[X_columns]
y = df['K_reduced']

train_df = df[df['mix'].isin(train_mix)]
test_df = df[df['mix'].isin(test_mix)]

X_train = train_df[X_columns]
y_train = train_df['K_reduced']
X_test = test_df[X_columns]
y_test = test_df['K_reduced']

# Create a logger that writes to "logs/run*":
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)

model = PySRRegressor(
    logger_spec=logger_spec,
    niterations=30000,
    populations = 48,
    unary_operators=["sqrt", "abs", "log"],
    parsimony=0.0000001,
    warmup_maxsize_by = 0.85,
    turbo=True,
)

model.fit(
    X_train, 
    y_train,
    )
