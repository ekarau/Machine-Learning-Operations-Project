import pytest
import os
from src.validation import validate_input, VAL_LOG_FILE

@pytest.fixture
def clean_log():
    if os.path.exists(VAL_LOG_FILE):
        os.remove(VAL_LOG_FILE)
    yield
    if os.path.exists(VAL_LOG_FILE):
        os.remove(VAL_LOG_FILE)

def test_validate_valid_input(clean_log):
    """Test with valid data."""
    data = {"Student_ID": "123", "Age": 25, "Progress_Percentage": 50, "Quiz_Score_Avg": 80}
    is_valid, msgs = validate_input(data)
    assert is_valid is True
    assert len(msgs) == 0

def test_validate_missing_field(clean_log):
    """Test missing required field."""
    data = {"Age": 25} # Missing others
    is_valid, msgs = validate_input(data)
    assert is_valid is False
    assert any("Missing required field" in m for m in msgs)

def test_validate_constraints(clean_log):
    """Test constraints (Age)."""
    data = {"Student_ID": "1", "Age": 200, "Progress_Percentage": 50, "Quiz_Score_Avg": 80}
    is_valid, msgs = validate_input(data)
    assert is_valid is False
    assert any("out of range" in m for m in msgs)
