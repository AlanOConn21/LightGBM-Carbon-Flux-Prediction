import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load all seven datasets
print("Loading and concatenating the datasets")
files = [
    "../databases/1.FI-Lom.csv",
    "../databases/2.GL-ZaF.csv",
    "../databases/3.IE-Cra.csv",
    "../databases/4.DE-Akm.csv",
    "../databases/5.FR-LGt.csv",
    "../databases/6.UK-AMo.csv",
    "../databases/7.SE-Htm.csv",
]
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True) # Concatenate them
print(f"Total rows: {len(df)}")

# Drop SITE_ID
df.drop(columns=['SITE_ID'], errors='ignore', inplace=True)
# Convert timestamp to usable format
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

# Create cyclical time features
df['hour'] = df['timestamp'].dt.hour
df['doy'] = df['timestamp'].dt.dayofyear
df['month'] = df['timestamp'].dt.month
df['hour_sin'] = np.sin(2*np.pi*df.hour/24)
df['hour_cos'] = np.cos(2*np.pi*df.hour/24)
df['doy_sin'] = np.sin(2*np.pi*df.doy/365)
df['doy_cos'] = np.cos(2*np.pi*df.doy/365)
df['month_sin'] = np.sin(2*np.pi*df.month/12)
df['month_cos'] = np.cos(2*np.pi*df.month/12)

# Check cyclical features are within range
for col in ['hour_sin','hour_cos','doy_sin','doy_cos','month_sin','month_cos']:
    assert df[col].between(-1,1).all(), f"{col} out of range [-1,1]"

# Create lag and rolling features 
lag_cols = ['LE_F_MDS','SW_IN_F','NETRAD','PPFD_IN','PPFD_OUT'] # Doesn't include NEE
lags = [1,3,6] 
windows = [3,6]
df.sort_values('timestamp', inplace=True)
for col in lag_cols:
    for l in lags:
        df[f'{col}_lag{l}'] = df[col].shift(l)
    for w in windows:
        r = df[col].rolling(window=w)
        df[f'{col}_roll{w}_mean'] = r.mean()
        df[f'{col}_roll{w}_std'] = r.std()
        df[f'{col}_roll{w}_min'] = r.min()
        df[f'{col}_roll{w}_max'] = r.max()

# Drop timestamp and rows with no NEE value as they are no longer useful
df.drop(columns=['timestamp'], inplace=True)
df.dropna(subset=['NEE_VUT_REF'], inplace=True)

# Split data 60/20/20
run = 1
train_val, test = train_test_split(df, test_size=0.20, random_state=run)
train, val = train_test_split(train_val, test_size=0.25, random_state=run)
# Drop rows without 6 hours of history in lagging columns
train.dropna(subset=[f'{c}_lag6' for c in lag_cols], inplace=True)
val.dropna(subset=[f'{c}_lag6' for c in lag_cols], inplace=True)

# Split features and target variable
TARGET = 'NEE_VUT_REF'
features = [c for c in train.columns if c != TARGET]
X_tr, y_tr = train[features], train[TARGET]
X_va, y_va = val[features], val[TARGET]
X_te, y_te = test[features], test[TARGET]

# Iterative Imputation
imp = IterativeImputer(random_state=run, max_iter=10)
X_tr = imp.fit_transform(X_tr)
X_va = imp.transform(X_va)
X_te = imp.transform(X_te)

# Standard Scaling
sc_X = StandardScaler()
X_tr = sc_X.fit_transform(X_tr)
X_va = sc_X.transform(X_va)
X_te = sc_X.transform(X_te)

sc_y = StandardScaler()
y_tr = sc_y.fit_transform(y_tr.values.reshape(-1,1)).ravel()
y_va = sc_y.transform(y_va.values.reshape(-1,1)).ravel()
y_te = sc_y.transform(y_te.values.reshape(-1,1)).ravel()

# Optuna hyperparameter optimisation
def objective(trial):
    params = {
        'objective':'huber','metric':'rmse','huber_delta':trial.suggest_float('huber_delta',0.1,10,log=True), # Uses Huber as an objective function
        'verbosity':-1,'boosting_type':'gbdt',
        'lambda_l1':trial.suggest_float('lambda_l1',1e-8,10,log=True),
        'lambda_l2':trial.suggest_float('lambda_l2',1e-8,10,log=True),
        'num_leaves':trial.suggest_int('num_leaves',31,256),
        'feature_fraction':trial.suggest_float('feature_fraction',0.5,1.0),
        'bagging_fraction':trial.suggest_float('bagging_fraction',0.5,1.0),
        'bagging_freq':trial.suggest_int('bagging_freq',1,10),
        'min_child_samples':trial.suggest_int('min_child_samples',5,100),
        'learning_rate':trial.suggest_float('learning_rate',1e-4,1e-1,log=True),
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtr)
    gbm = lgb.train(params, dtr, num_boost_round=5000,
                    valid_sets=[dtr,dval],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pred_va = gbm.predict(X_va, num_iteration=gbm.best_iteration)
    return np.sqrt(mean_squared_error(y_va, pred_va))

study = optuna.create_study(direction='minimize', pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=10)) # Pruner for efficiency
study.optimize(objective, n_trials=100, timeout=60*20)  # Timeout of 20 minutes
best = study.best_params
print("Best parameters:", best)

# Final training on training + validation data
best.update({'objective':'huber','metric':'rmse','verbosity':-1})
dall = lgb.Dataset(np.vstack([X_tr, X_va]), label=np.hstack([y_tr, y_va]))
final = lgb.train(best, dall, num_boost_round=5000, valid_sets=[dall], callbacks=[lgb.early_stopping(50)])

# Predictions are made
pred_s = final.predict(X_te, num_iteration=final.best_iteration)
pred = sc_y.inverse_transform(pred_s.reshape(-1,1)).ravel()
true = sc_y.inverse_transform(y_te.reshape(-1,1)).ravel()
mask = (~np.isnan(true)) & (~np.isnan(pred))
true, pred = true[mask], pred[mask]

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(true, pred))
r2 = r2_score(true, pred)
print(f"TEST RMSE: {rmse:.4f},  R²: {r2:.4f}")

# Create scatter plot of actual vs predicted NEE
plt.figure(figsize=(6,6))
plt.scatter(true, pred, alpha=0.4)
mn, mx = true.min(), true.max()
plt.plot([mn,mx], [mn,mx], 'r--')
plt.xlabel("Actual NEE")
plt.ylabel("Predicted NEE")
plt.title("Actual vs Predicted NEE (Single Run)")
plt.tight_layout()
plt.show()

# Create plot of 20 most important features
gain = final.feature_importance(importance_type='gain')
fi = pd.DataFrame({'feature': features, 'gain': gain}).sort_values('gain', ascending=False).head(20)

plt.figure(figsize=(10,6))
sns.barplot(x='gain', y='feature', data=fi, palette='viridis')
plt.title("Top 20 LightGBM Features by Gain")
plt.tight_layout()
plt.show()
