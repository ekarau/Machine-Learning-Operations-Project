import pytest
from src.fallback import HeuristicModel

def test_heuristic_high_probability():
    """Test high engagement leads to completion prediction."""
    model = HeuristicModel()
    data = {"Progress_Percentage": 95, "Quiz_Score_Avg": 80}
    result = model.predict(data)
    assert result["prediction"] == 1
    assert result["probability"] > 0.9

def test_heuristic_low_probability():
    """Test low engagement leads to non-completion."""
    model = HeuristicModel()
    data = {"Progress_Percentage": 20, "Quiz_Score_Avg": 40}
    result = model.predict(data)
    assert result["prediction"] == 0
    assert result["probability"] < 0.2

def test_heuristic_missing_data():
    """Test graceful handling of missing data (defaults to 0)."""
    model = HeuristicModel()
    data = {} 
    result = model.predict(data)
    assert result["status"] == "fallback_heuristic"
    assert result["prediction"] == 0
