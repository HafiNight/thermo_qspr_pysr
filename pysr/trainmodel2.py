import pandas as pd
import numpy as np
import pickle
from pysr import PySRRegressor, TensorBoardLoggerSpec, TemplateExpressionSpec
from sklearn.model_selection import train_test_split

# Import data
with open('../data/df_filtered_reduced.pkl', 'rb') as f:
    df = pickle.load(f)

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
    niterations= 20000,
    populations = 48,
    maxsize = 40,
    unary_operators=["sqrt", "abs", "log"],
    parsimony=0.0000001,
    warmup_maxsize_by = 0.85,
    turbo=True,
    nested_constraints = {"sqrt":{"sqrt":0},"log":{"log":0}},
    dimensional_constraint_penalty = 10**5,
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 20"
        # Stop early if we find a good and simple equation
    ),
    warm_start=True,
)


# Function to randomly switch columns for some rows
def switch_columns_randomly(df, fraction=0.1):
    """Randomly switch mA with mB, KA with KB, and Tr_A with Tr_B for a fraction of the rows."""
    # Randomly choose a fraction of the rows to swap columns
    rows_to_switch = np.random.choice(df.index, size=int(len(df) * fraction), replace=False)
    
    # Create a copy of the dataframe to avoid modifying the original
    df_switched = df.copy()
    
    # Switch columns for the chosen rows
    df_switched.loc[rows_to_switch, 'mA'], df_switched.loc[rows_to_switch, 'mB'] = df_switched.loc[rows_to_switch, 'mB'], df_switched.loc[rows_to_switch, 'mA']
    df_switched.loc[rows_to_switch, 'KA'], df_switched.loc[rows_to_switch, 'KB'] = df_switched.loc[rows_to_switch, 'KB'], df_switched.loc[rows_to_switch, 'KA']
    df_switched.loc[rows_to_switch, 'Tr_A'], df_switched.loc[rows_to_switch, 'Tr_B'] = df_switched.loc[rows_to_switch, 'Tr_B'], df_switched.loc[rows_to_switch, 'Tr_A']
    
    return df_switched

# Train many times with switching columns randomly for some rows each time
for i in range(5):
    print(f"Training iteration {i + 1}...")
    
    # Randomly switch columns for some rows in the training and test sets
    train_df_new = switch_columns_randomly(train_df, fraction=0.25*i)  # % of rows randomly switched
    test_df_new = switch_columns_randomly(test_df, fraction=0.25*i)    # % of rows randomly switched
    
    # Update X_train and y_train for the new switched columns
    X_train_new = train_df_new[X_columns]
    y_train_new = train_df_new['K_reduced']
    X_test_new = test_df_new[X_columns]
    y_test_new = test_df_new['K_reduced']
    
    # Train model with current setup
    model.fit(
        X_train_new, 
        y_train_new,
        X_units=["1", "1", "W*m^-1*K^-1", "W*m^-1*K^-1"],
        y_units="W*m^-1*K^-1"
    )
