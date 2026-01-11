
import pytest
import json
import os
import sys

# Ensure src is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.version import __version__

# Point to root directory where train_model.py saves it
METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_metrics.json")

@pytest.fixture(scope="module")
def model_metrics():
    """
    Load the actual training metrics from model_metrics.json.
    This enables 'Real Data' testing instead of Mocks.
    """
    if not os.path.exists(METRICS_PATH):
        pytest.fail(f"Artifact {METRICS_PATH} not found. Run 'python train_model.py' before running tests.")
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return metrics

def test_accuracy_threshold_gate(model_metrics):
    """
    Quality Gate: Validates that the best model meets the accuracy threshold.
    Uses REAL data from training artifacts.
    """
    # Find the best model (usually the one saved as 'model.pkl' which is XGBoost in train_model.py)
    # Or just check if ANY model passed.
    
    # Let's pick the Production candidate (XGBoost)
    prod_candidate = next((m for m in model_metrics if "XGBoost" in m.get("Model", "")), None)
    
    if not prod_candidate:
        # Fallback to the one with highest accuracy if specific name not found
        prod_candidate = max(model_metrics, key=lambda x: x.get("Accuracy", 0))

    threshold = float(os.getenv("MODEL_ACCURACY_THRESHOLD", 0.50))
    current_accuracy = prod_candidate.get("Accuracy", 0)
    
    print(f"Checking Model: {prod_candidate['Model']} with Accuracy: {current_accuracy}")
    assert current_accuracy >= threshold, \
        f"Model accuracy {current_accuracy} is below threshold {threshold}"

def test_rebalancing_impact(model_metrics):
    """
    Verifies that the rebalancing strategy actually improved Recall on the minority class
    compared to the Baseline (Unbalanced) model.
    """
    baseline = next((m for m in model_metrics if m["Model"] == "Baseline_RF"), None)
    balanced = next((m for m in model_metrics if "XGBoost" in m["Model"] or "RandomForest" in m["Model"] and m["Model"] != "Baseline_RF"), None)

    assert baseline is not None, "Baseline metrics not found in artifacts"
    assert balanced is not None, "Balanced model metrics not found"

    print(f"Baseline Recall: {baseline.get('Recall', 0)}")
    print(f"Balanced Recall: {balanced.get('Recall', 0)}")

    # We expect some improvement or at least maintenance of recall while having good accuracy
    # In many imbalanced cases, Baseline Recall is very low (e.g. 0.3) and Balanced is higher (0.7).
    
    assert balanced['Recall'] > baseline['Recall'], \
        f"Rebalancing failed to improve Recall. Baseline: {baseline['Recall']}, New: {balanced['Recall']}"

def test_model_version_consistency(model_metrics):

    # Get version from artifacts
    prod_candidate = model_metrics[0] # Just check the first one or all
    artifact_version = prod_candidate.get("Version")
    
    assert artifact_version is not None, "Version tag is missing in model metrics"
    
    # Ensure it matches the code version
    assert artifact_version == __version__, \
        f"Artifact version {artifact_version} mismatch with Code version {__version__}"

    # Semantic Versioning Check (Simple)
    parts = artifact_version.split('.')
    assert len(parts) >= 3, "Version does not follow SemVer format (x.y.z)"
