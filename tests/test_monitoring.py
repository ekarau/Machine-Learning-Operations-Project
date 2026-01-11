import json
import os
import shutil
import pytest
from src.monitoring import check_drift, LOG_FILE

# Use a temporary log file for testing
TEST_LOG_FILE = 'models/logs/test_prediction_logs.jsonl'

@pytest.fixture
def mock_log_file(monkeypatch):
    """Mocks the LOG_FILE path to a test file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(TEST_LOG_FILE), exist_ok=True)
    
    # Patch the module-level variable
    monkeypatch.setattr('src.monitoring.LOG_FILE', TEST_LOG_FILE)
    
    # Clean up before test
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
        
    yield
    
    # Clean up after test
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)

def test_check_drift_no_logs(mock_log_file):
    """Test drift check when no log file exists."""
    # Ensure file is gone
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
        
    result = check_drift()
    assert result['status'] == 'no_logs'

def test_check_drift_empty_logs(mock_log_file):
    """Test drift check with empty log file."""
    with open(TEST_LOG_FILE, 'w') as f:
        pass
        
    result = check_drift()
    assert result['status'] == 'empty_logs'

def test_check_drift_no_drift(mock_log_file):
    """Test drift check with normal data (mean ~0.5)."""
    with open(TEST_LOG_FILE, 'w') as f:
        for _ in range(100):
            entry = {"probability": 0.55}
            f.write(json.dumps(entry) + '\n')
            
    result = check_drift()
    assert result['status'] == 'success'
    assert result['drift_detected'] is False
    assert 0.5 <= result['current_mean'] <= 0.6

def test_check_drift_detected(mock_log_file):
    """Test drift check when drift is high (mean ~0.9)."""
    with open(TEST_LOG_FILE, 'w') as f:
        for _ in range(100):
            entry = {"probability": 0.9}
            f.write(json.dumps(entry) + '\n')
            
    result = check_drift()
    assert result['status'] == 'success'
    assert result['drift_detected'] is True
    assert result['current_mean'] > 0.8

def test_check_feature_drift_detected(mock_log_file):
    """Test feature input drift detection."""
    from src.monitoring import check_feature_drift
    with open(TEST_LOG_FILE, 'w') as f:
        for _ in range(100):
            # Input Age = 50, Baseline is 25 -> Drift!
            entry = {"input": {"Age": 50}, "probability": 0.5, "prediction": 1}
            f.write(json.dumps(entry) + '\n')
            
    result = check_feature_drift(feature_name='Age')
    assert result['status'] == 'success'
    assert result['drift_detected'] is True
    assert result['current_mean'] == 50.0

def test_check_feature_drift_no_drift(mock_log_file):
    """Test feature input no drift."""
    from src.monitoring import check_feature_drift
    
    with open(TEST_LOG_FILE, 'w') as f:
        for _ in range(100):
            # Input Age = 25, Baseline is 25 -> No Drift
            entry = {"input": {"Age": 25}, "probability": 0.5, "prediction": 1}
            f.write(json.dumps(entry) + '\n')
            
    result = check_feature_drift(feature_name='Age')
    assert result['status'] == 'success'
    assert result['drift_detected'] is False
