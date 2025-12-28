from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add /opt/airflow to the path so the src module can be found inside Docker
sys.path.append('/opt/airflow')

# Import your custom modules
from src.ingest import load_data
from src.validate import validate_input_data
from src.preprocess import clean_data, split_data, balance_data
from src.features import apply_feature_cross, apply_hashing

# File Paths (Using interim steps to create a visual chain in Airflow)
DATA_DIR = '/opt/airflow/data'
RAW_PATH = f'{DATA_DIR}/raw/Course_Completion_Prediction.csv'
STAGE_1_VALIDATED = f'{DATA_DIR}/interim/1_validated.csv'
STAGE_2_CLEANED = f'{DATA_DIR}/interim/2_cleaned.csv'
STAGE_3_FEATURES = f'{DATA_DIR}/interim/3_features.csv'
PROCESSED_PATH = f'{DATA_DIR}/processed'

# Create necessary directories if they don't exist
os.makedirs(f'{DATA_DIR}/interim', exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Task 1: Ingest and Validate Data
def task_ingest_validate():
    print("--- STEP 1: Ingest & Validate ---")
    df = load_data(RAW_PATH)
    
    # Standardize column names (remove hidden spaces)
    df.columns = df.columns.str.strip()
    
    # Validate data schema
    df = validate_input_data(df)
    
    # Save for the next step
    df.to_csv(STAGE_1_VALIDATED, index=False)
    print(f"Validated data saved to {STAGE_1_VALIDATED}")

# Task 2: Clean Data (Preprocessing)
def task_clean():
    print("--- STEP 2: Cleaning ---")
    df = pd.read_csv(STAGE_1_VALIDATED)
    
    # Apply cleaning logic
    df = clean_data(df)
    
    # Save cleaned data
    df.to_csv(STAGE_2_CLEANED, index=False)
    print(f"Cleaned data saved to {STAGE_2_CLEANED}")

# Task 3: Feature Engineering
def task_feature_eng():
    print("--- STEP 3: Feature Engineering ---")
    df = pd.read_csv(STAGE_2_CLEANED)
    
    # Apply Feature Crossing
    df = apply_feature_cross(df)
    
    # Apply Hashing (e.g., for High Cardinality ID columns)
    # Note: Applying to the full dataset here for pipeline simplicity
    df = apply_hashing(df, 'Student_ID', n_features=100)
    
    # Save feature-engineered data
    df.to_csv(STAGE_3_FEATURES, index=False)
    print(f"Feature engineered data saved to {STAGE_3_FEATURES}")

# Task 4: Split, Balance, and Final Save
def task_split_balance_save():
    print("--- STEP 4: Split, Balance & Save ---")
    df = pd.read_csv(STAGE_3_FEATURES)
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Merge back to DataFrames for easier handling
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Balance Data (Only applied to Training Set to prevent leakage)
    train_df = balance_data(train_df)
    
    # Save Final Artifacts
    train_df.to_csv(f'{PROCESSED_PATH}/train_processed.csv', index=False)
    test_df.to_csv(f'{PROCESSED_PATH}/test_processed.csv', index=False)
    print("Pipeline completed successfully. Final artifacts saved.")

# DAG Configuration
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_term_project_pipeline',
    default_args=default_args,
    description='MLOps End-to-End Data Engineering Pipeline',
    schedule='@once',  # Runs once when triggered
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'etl', 'group_project']
) as dag:

    # Define Tasks
    t1 = PythonOperator(
        task_id='1_ingest_and_validate',
        python_callable=task_ingest_validate
    )

    t2 = PythonOperator(
        task_id='2_clean_data',
        python_callable=task_clean
    )

    t3 = PythonOperator(
        task_id='3_feature_engineering',
        python_callable=task_feature_eng
    )

    t4 = PythonOperator(
        task_id='4_split_balance_save',
        python_callable=task_split_balance_save
    )

    # Define Dependencies (Linear Chain)
    # This creates the visual flow: t1 -> t2 -> t3 -> t4
    t1 >> t2 >> t3 >> t4

if __name__ == "__main__":
    print("🚀 Manual Execution Started for Testing...")
    task_ingest_validate()
    task_clean()
    task_feature_eng()
    task_split_balance_save()