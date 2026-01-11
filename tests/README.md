# 🧪 MLOps Project - Test Suite Documentation

This directory contains the comprehensive automated test suite for the component, meeting the MLOps "Continuous Testing" and "Automated Acceptance Gate" requirements.

## 📂 Test Structure

### 1. Compliance & Requirements
*   **`test_compliance_requirements.py`**: **[CRITICAL]** Explicitly tests mandatory project requirements such as:
    *   **Feature Engineering Logic:** Verifying deterministic behavior.
    *   **Database Interactions:** Mock testing of query construction.
    *   **Hashing Collision Rates:** Validating the high-cardinality strategy.

### 2. Unit & Functional Tests
*   **`test_app.py`**: Tests FastAPI endpoints (`/health`, `/predict`) for correct status codes, JSON responses, and error handling.
*   **`test_feature_engineering.py`**: Validates specific feature engineering functions like `clean_data` and hashing.
*   **`test_transformers.py`**: Unit tests for custom Scikit-Learn transformers (`HashingTransformer`, `FeatureCrossTransformer`).
*   **`test_validation.py`**: Checks Input Pydantic schemas and data validation rules.

### 3. Integration & E2E Tests
*   **`test_e2e.py`**: End-to-End test simulating the full flow: Data Loading -> Training -> Model Saving -> Prediction.
*   **`test_fallback.py`**: Verifies the **Algorithmic Fallback** mechanism (switching to Heuristic model when the main model fails).

### 4. Quality Assurance
*   **`test_data_quality.py`**: Checks for missing values, schema conformance, and statistical distributions.
*   **`test_model_quality.py`**: Load the trained model and verifies performance metrics (Accuracy > Threshold) on a test set.

### 5. Infrastructure & Operations
*   **`test_docker_smoke.py`**: **[Smoke Test]** Verifies that the Docker container starts correctly, exposes port 8000, and responds to HTTP requests.
*   **`test_monitoring.py`**: Checks if Prometheus metrics (e.g., `prediction_counter`) are correctly exposed and incremented.

### 6. Performance / Load Testing
*   **`locustfile.py`**: A Locust script for load testing the API to ensure it handles concurrent requests under SLA.

---

## 🚀 How to Run Tests

### 1. Unit & Integration (Fast)
Run these commands locally during development:

```powershell
# Run everything
pytest

# Run specific file
pytest tests/test_compliance_requirements.py
```

### 2. Docker Smoke Test (Acceptance)
This test requires the Docker container to be running.

```powershell
# 1. Build and Run Container
docker-compose up -d api

# 2. Run Smoke Test
pytest tests/test_docker_smoke.py
```

### 3. Load Testing
```powershell
# Install Locust
pip install locust

# Start Swarm
locust -f tests/locustfile.py
```
*Access interface at http://localhost:8089*

---

## 🤖 CI/CD Integration (GitLab CI)

These tests are automatically executed in the **GitLab CI/CD Pipeline**:
*   **Test Stage:** Runs `pytest` (Unit + Integration)
*   **Security Stage:** Runs `bandit` (Code Security)
*   **Acceptance Stage:** Runs `docker_build` and `acceptance_smoke` to verify the container image.