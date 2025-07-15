import pandas as pd
import numpy as np
import pickle
from pysr import PySRRegressor, TensorBoardLoggerSpec, TemplateExpressionSpec
from sklearn.model_selection import train_test_split

# Import data
with open('data/df_tc.pkl', 'rb') as f:
    df = pickle.load(f)

df['mix'] = df['compA'] + df['compB']
mixtures = df['mix'].unique()

train_mix, test_mix = train_test_split(mixtures, test_size=0.3, random_state=42)

df = df[df['P'] != 77000]
X_columns = ['XA', 'XB', 'KA', 'KB', 'Tr_A', 'Tr_B','Tr_mix']
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

# Define template:
template = TemplateExpressionSpec(
    combine="f(XA, KA) + g(XB, KB) + p1[1] * h(Tr_A, Tr_B, Tr_mix, XA, XB) + p2[1]",
    expressions=["f","g" ,"h"],
    variable_names=["XA", "XB", "KA", "KB", "Tr_A", "Tr_B", "Tr_mix"],
    parameters={"p1": 1, "p2": 1},
)

model = PySRRegressor(
    logger_spec=logger_spec,
    niterations=2000,
    binary_operators=["+", "*", "/", "-"],
    unary_operators=["log"],
    parsimony=0.000001,
    turbo=True,
    warmup_maxsize_by=0.75,
    expression_spec=template,
)

model.fit(
    X_train, 
    y_train
    )