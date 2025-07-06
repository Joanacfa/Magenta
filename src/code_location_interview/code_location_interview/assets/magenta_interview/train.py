import logging
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from dagster import AssetOut, AssetIn, asset, get_dagster_logger, multi_asset, file_relative_path

from sklearn.base import BaseEstimator, TransformerMixin

# from dagstermill import define_dagstermill_asset

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "training"

@multi_asset(
     group_name=group_name,
     outs={
         "train_data": AssetOut(),
         "test_data": AssetOut(),
     },
 )
def split_train_test(df_input_preprocessed):
    logger.info("Split train and test dataset")

    X = df_input_preprocessed.drop(columns=['rating_account_id', 'customer_id', 'has_done_upselling'])
    y = df_input_preprocessed['has_done_upselling']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    return train_data, test_data
 
 
@asset(group_name=group_name)
def classifier(train_data):
    X_train, y_train = train_data

# Transform categorical features
    categorical_features = ['smartphone_brand']
    ohe_transformer = OneHotEncoder(handle_unknown='ignore')

# Impute with 0
    impute_with_zero_features = ['prolongation_n', 'asked_prolongation', 'rechnungsanfragen_n', 'asked_rechnungsanfragen', 'produkte&services-tarifwechsel_n', 'asked_produkte&services-tarifwechsel',
       'produkte&services-tarifdetails_n', 'asked_produkte&services-tarifdetails', 'available_gb']
    impute_with_zero_transformer = SimpleImputer(strategy='constant', fill_value=0)

# Impute with -1
    impute_with_minus_one_features = ['days_since_last_prolongation', 'days_since_last_rechnungsanfragen', 'days_since_last_produkte&services-tarifwechsel', 'days_since_last_produkte&services-tarifdetails']
    impute_with_minus_one_transformer = SimpleImputer(strategy='constant', fill_value=-1)

    object_features = ['available_gb']

    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ohe_transformer, categorical_features),
        ('impute_0', impute_with_zero_transformer, impute_with_zero_features),
        ('impute_minus_1', impute_with_minus_one_transformer, impute_with_minus_one_features),
        ('drop_col', 'drop', categorical_features),
        ('drop_col2', 'drop', object_features),
    ]
)

    xgboost_classifier = XGBClassifier(
         n_estimators=100,
         objective='binary:logistic',
         scale_pos_weight=13,
         max_delta_step=1,
         random_state=42,
         learning_rate=0.01,
         max_depth=2,
         gamma=0,
         reg_lambda=1,    
         reg_alpha=0  
        )

    
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgboost_classifier)
])
    
    pipeline.fit(X_train, y_train)
    return pipeline