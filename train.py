import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle
import xgboost as xgb
from collections import Counter  # If used for word frequency analysis
from itertools import combinations  # If combinations are still relevant
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Parameters


eta = 0.1
# max_depth = 6
# gamma = 0.1
# subsample = 0.8
# colsample_bytree = 0.8
# min_child_weight = 15
nthread = -1
seed = 1
verbosity = 1
eval_metric = 'rmse'
# output_file = f'model_eta={eta}_maxdepth={max_depth}_gamma={gamma}_subsample={subsample}_colsample_bytree={colsample_bytree}_minchild={min_child_weight}.bin'
output_file = "model.bin"
n_estimators =  445
learning_rate = 0.01130632861819067
max_depth = 4
min_child_weight = 9
gamma = 1.8995562787677835
subsample =  0.8277050661510729
colsample_bytree = 0.6877153806619976
reg_alpha =  3.3716073542292255
reg_lambda = 2.9307937516750493


#import the data
# url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv('data/Train.csv')
df = pd.DataFrame(data)

#####################
#####################
# data preparation









#####################
#####################

# Splitting the dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state = 1)
df_train, df_val= train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)

y_train = (df_train.BeatsPerMinute ).astype('int').values
y_val = (df_val.BeatsPerMinute ).astype('int').values
y_test = (df_test.BeatsPerMinute).astype('int').values

del df_train['BeatsPerMinute']
del df_test['BeatsPerMinute']
del df_val['BeatsPerMinute']


dv = DictVectorizer(sparse=True)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.to_dict(orient='records')
X_val= dv.fit_transform(val_dicts)

# Training function
def train(df_train, y_train, params):
    train_dict = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    model = xgb.train(params, dtrain, num_boost_round=168, evals=[(dtrain, 'train'),], verbose_eval=5, early_stopping_rounds=5)

    return dv, model

# Prediction function
def predict(df, dv, model):
    data_dict = df.to_dict(orient='records')
    X = dv.transform(data_dict)
    features = list(dv.get_feature_names_out())  # Ensure it's a list
    dmatrix = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dmatrix)
    return y_pred

# XGBoost parameters
xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'gamma': gamma,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'min_child_weight': min_child_weight,
    'nthread': nthread,
    'seed': seed,
    'verbosity': verbosity,
    'eval_metric': eval_metric

}
# Training the final model
print('Training the final model...')
dv, model = train(df_train, y_train, xgb_params)
y_pred = predict(df_val, dv, model)
rmse = mean_squared_error(y_val, y_pred)
print(f'Validation RMSE: {rmse:.3f}')

# Training the final model
print('Training the final model...')
dv, model = train(df_train, y_train, xgb_params)

# Test set evaluation
y_pred = predict(df_test, dv, model)
rmse = mean_squared_error(y_test, y_pred)
print(f'Test RMSE: {rmse:.3f}')

# Save the final model and DictVectorizer
output_file = f'xgboost_model{eta}_{max_depth}_{gamma}_{subsample}_{colsample_bytree}_{min_child_weight}.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model and DictVectorizer are saved to {output_file}')