from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# When Airflow runs inside Docker, /opt/airflow is the root directory.
# We add it to the path so the src module can be found.
sys.path.append('/opt/airflow')

from src.ingest import load_data
from src.validate import validate_input_data
from src.preprocess import clean_data, split_data, balance_data
from src.features import apply_feature_cross, apply_hashing

# File Paths (Paths inside Docker)
RAW_DATA_PATH = '/opt/airflow/data/raw/Course_Completion_Prediction.csv'
PROCESSED_DATA_PATH = '/opt/airflow/data/processed'

def run_etl_pipeline():
    # 1. Ingest (Data Loading)
    df = load_data(RAW_DATA_PATH)
    
    # 2. Preprocess (Cleaning)
    df = clean_data(df)
    df = validate_input_data(df)

    # 3. Feature Engineering - Cross (Feature Crossing)
    df = apply_feature_cross(df)
    
    # 4. Split (Splitting)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Merging as DataFrame (For ease of processing)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # 5. Rebalancing (Applied only to Training Data!)
    train_df = balance_data(train_df)
    
    # 6. Hashing (High Cardinality - Student_ID)
    # The transformation logic applied to the Train set is also applied to the Test set
    train_df = apply_hashing(train_df, 'Student_ID', n_features=100)
    test_df = apply_hashing(test_df, 'Student_ID', n_features=100)
    
    # 7. Saving
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    train_df.to_csv(f'{PROCESSED_DATA_PATH}/train_processed.csv', index=False)
    test_df.to_csv(f'{PROCESSED_DATA_PATH}/test_processed.csv', index=False)
    
    print("Pipeline completed successfully. Files saved.")

# DAG Settings
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_term_project_pipeline',
    default_args=default_args,
    description='MLOps Data Engineering Pipeline',
    schedule_interval='@once',  # Or '@daily'
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    run_etl = PythonOperator(
        task_id='run_full_etl',
        python_callable=run_etl_pipeline
    )