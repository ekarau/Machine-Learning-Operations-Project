
import pytest
import pandas as pd
import os
import shutil
import joblib
from train_model import MLEngineerPipeline

@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        'Student_ID': ['S1', 'S2', 'S3', 'S4'],
        'Category': ['A', 'B', 'A', 'B'],
        'Course_Level': ['L1', 'L2', 'L1', 'L2'],
        'Enrollment_Date': ['01-01-2023', '02-02-2023', '03-03-2023', '04-04-2023'],
        'Progress_Percentage': [10, 20, 90, 100],
        'Completed': ['Not Completed', 'Not Completed', 'Completed', 'Completed'],
        # Extra columns to simulate realistic data
        'Age': [20, 21, 22, 23],
        'Instructor_Rating': [4.5, 4.0, 3.5, 5.0]
    })

def test_pipeline_e2e(tmp_path, dummy_data):
    # Setup
    os.chdir(tmp_path) # Work in temp dir
    
    # Run Pipeline
    pipeline = MLEngineerPipeline(dummy_data, experiment_name="Test_Exp")
    pipeline.run_classification_experiments()
    
    # Assertions
    assert os.path.exists("checkpoints")
    assert os.path.exists("models/model.pkl")
    
    # Verify Model Loading
    loaded_pipe = joblib.load("models/model.pkl")
    assert loaded_pipe is not None
    
    # Verify Prediction
    pred = loaded_pipe.predict(dummy_data.iloc[[0]])
    assert pred is not None
