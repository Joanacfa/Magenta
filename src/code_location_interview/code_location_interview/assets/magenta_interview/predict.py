import logging
import sys
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dagster import asset, get_dagster_logger


log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "predict"

@asset(
     group_name=group_name,
     
 )
def predictions(classifier, df_input_preprocessed):
     X = df_input_preprocessed.drop(columns=['has_done_upselling'])
     y = df_input_preprocessed['has_done_upselling']
     X_train, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Test predictions
     X_test = X_test_df.drop(columns=['rating_account_id', 'customer_id'])

     y_pred = classifier.predict(X_test)
     
     predictions_df = pd.DataFrame({
        'rating_account_id': X_test_df['rating_account_id'],
        'customer_id': X_test_df['customer_id'],
        'available_gb': X_test_df['available_gb'],
        'gross_mrc': X_test_df['gross_mrc'],
        'y_predicted': y_pred,
        'y':y_test
    })
     
     return predictions_df