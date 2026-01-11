
import sys
import os
import pytest
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock TensorFlow and MLflow to allow testing without heavy dependencies
# This is critical if the CI environment doesn't have them installed.
# We do this BEFORE any test imports src modules.

class MockTensorFlow(MagicMock):
    pass

class MockKeras(MagicMock):
    pass
    
class MockMLflow(MagicMock):
    pass

# Helper to create a functional mock for simple imports
mock_tf = MockTensorFlow()
mock_tf.keras.models.load_model = MagicMock(return_value=MagicMock(predict=lambda x: [[0.8]]))
# Mock metrics
mock_tf.keras.metrics.AUC = MagicMock()

mock_mlflow = MockMLflow()

# Mock psycopg2
mock_psycopg2 = MagicMock()
mock_conn = MagicMock()
mock_conn.closed = 0
mock_conn.cursor.return_value.fetchone.return_value = [1]
mock_psycopg2.connect.return_value = mock_conn



# Patch sys.modules
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()
sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.tensorflow'] = MagicMock()
sys.modules['psycopg2'] = mock_psycopg2

# Mock submodules to prevent "mlflow is not a package" error
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['mlflow.xgboost'] = MagicMock()


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure environment is clean between tests."""
    # We could reset mocks here if needed
    yield
