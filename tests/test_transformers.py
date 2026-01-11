
import pytest
import pandas as pd
import numpy as np
from src.pipeline_transformers import CleanDataTransformer, FeatureCrossTransformer, HashingTransformer

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'Student_ID': ['S1', 'S2'],
        'Category': ['A', 'B'],
        'Course_Level': ['L1', 'L2'],
        'Enrollment_Date': ['01-01-2023', '02-02-2023'],
        'Other': [1, 2]
    })

def test_clean_data_transformer(mock_df):
    cleaner = CleanDataTransformer()
    res = cleaner.transform(mock_df)
    
    assert 'Enrollment_Date' not in res.columns
    assert 'Enrollment_Month' in res.columns
    assert res.iloc[0]['Enrollment_Month'] == 1
    # Ensure other columns persist
    assert 'Category' in res.columns

def test_feature_cross_transformer(mock_df):
    crosser = FeatureCrossTransformer(col1='Category', col2='Course_Level', new_col_name='Crossed')
    res = crosser.transform(mock_df)
    
    assert 'Crossed' in res.columns
    assert res.iloc[0]['Crossed'] == 'A_L1'
    assert res.iloc[1]['Crossed'] == 'B_L2'

def test_hashing_transformer(mock_df):
    hasher = HashingTransformer(col_name='Student_ID', n_features=5)
    res = hasher.transform(mock_df)
    
    assert 'Student_ID' not in res.columns
    # Check for hashed columns
    hashed_cols = [c for c in res.columns if 'hashed_Student_ID_' in c]
    assert len(hashed_cols) == 5
