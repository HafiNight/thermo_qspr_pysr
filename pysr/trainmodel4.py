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
    niterations=2000,
    unary_operators=["abs"],
    dimensional_constraint_penalty=10**5,
    parsimony=0.000001,
    turbo=True,
    warm_start=True,
    warmup_maxsize_by=0.75,
)
#removed weight optimize

def switch_columns_duplicate(df):
    """Duplicate df
    Switch XA and XB; mA and mB; KA and KB; Tr_A and Tr_B
    Merge the two dataframes
    """
    df_duplicate = df.copy()
    df_duplicate[['XA', 'XB']] = df_duplicate[['XB', 'XA']].values
    df_duplicate[['mA', 'mB']] = df_duplicate[['mB', 'mA']].values
    df_duplicate[['KA', 'KB']] = df_duplicate[['KB', 'KA']].values
    df_duplicate[['Tr_A', 'Tr_B']] = df_duplicate[['Tr_B', 'Tr_A']].values
    return pd.concat([df, df_duplicate])

train_df_new = switch_columns_duplicate(train_df) 

# Update X_train and y_train for the new switched columns
X_train_new = train_df_new[X_columns]
y_train_new = train_df_new['K']

# Train model with current setup
model.fit(
    X_train_new, 
    y_train_new,
    X_units=["1", "1", "W*m^-1*K^-1", "W*m^-1*K^-1"],
    y_units="W*m^-1*K^-1"
)