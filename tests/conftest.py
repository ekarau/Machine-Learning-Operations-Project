import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# --------------------------------------------------
# Ensure project root is on PYTHONPATH (CI-safe)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Mock heavy dependencies (TensorFlow, MLflow, DB)
# --------------------------------------------------
class MockTensorFlow(MagicMock):
    pass

class MockMLflow(MagicMock):
    pass

mock_tf = MockTensorFlow()
mock_tf.keras.models.load_model = MagicMock(
    return_value=MagicMock(predict=lambda x: [[0.8]])
)
mock_tf.keras.metrics.AUC = MagicMock()

mock_mlflow = MockMLflow()

mock_psycopg2 = MagicMock()
mock_conn = MagicMock()
mock_conn.closed = 0
mock_conn.cursor.return_value.fetchone.return_value = [1]
mock_psycopg2.connect.return_value = mock_conn

sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()

sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.tensorflow'] = MagicMock()
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['mlflow.xgboost'] = MagicMock()

sys.modules['psycopg2'] = mock_psycopg2

# --------------------------------------------------
# Pytest fixtures
# --------------------------------------------------
@pytest.fixture(autouse=True)
def clean_env():
    yield
