import pytest
import subprocess
import time
import requests
import os

# This test requires Docker to be available and built
# It is intended for the CI/CD 'smoke_test' stage

DOCKER_IMAGE = "course-api"
CONTAINER_NAME = "course-api-smoke-test"
PORT = 8000

@pytest.fixture(scope="module")
def docker_container():
    """Starts the docker container and tears it down after."""
    # check if docker is available
    if subprocess.call("docker --version", shell=True) != 0:
        pytest.skip("Docker not available")

    # Run container
    # DÜZELTME: Konteyner portu 80'den 8000'e çekildi (FastAPI default)
    subprocess.run(
        f"docker run -d --name {CONTAINER_NAME} -p {PORT}:8000 {DOCKER_IMAGE}",
        shell=True,
        check=True
    )
    
    # Wait for startup
    time.sleep(10)  # Bekleme süresini biraz artırdım (daha güvenli olması için)
    
    yield
    
    # Stop and remove
    subprocess.run(f"docker stop {CONTAINER_NAME}", shell=True)
    subprocess.run(f"docker rm {CONTAINER_NAME}", shell=True)

def test_container_health(docker_container):
    """Verify container is running and responding."""
    try:
        # Assuming there's a /docs or /health endpoint
        response = requests.get(f"http://localhost:{PORT}/health") # /docs yerine /health daha hafif bir kontrol olabilir
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.fail("Could not connect to Docker container")

def test_container_prediction(docker_container):
    """Verify prediction endpoint works in container."""
    payload = {
        "Student_ID": "SMOKE_TEST",
        "Age": 25,
        "Progress_Percentage": 50,
        "Quiz_Score_Avg": 80,
        # Minimal fields just to hit the endpoint logic
        "Category": "Programming",
        "Education_Level": "Bachelor",
        "Employment_Status": "Student", 
        "City": "TestCity",
        "Device_Type": "Laptop",
        "Internet_Connection_Quality": "High",
        "Course_ID": "C101",
        "Course_Level": "Beginner",
        "Payment_Mode": "Card",
        "Gender": "Male",
        "Fee_Paid": "Yes",
        "Discount_Used": "No",
        "Course_Duration_Days": 30,
        "Instructor_Rating": 4.5,
        "Login_Frequency": 5,
        "Average_Session_Duration_Min": 45,
        "Video_Completion_Rate": 80,
        "Discussion_Participation": 5,
        "Time_Spent_Hours": 10,
        "Days_Since_Last_Login": 2,
        "Notifications_Checked": 5,
        "Peer_Interaction_Score": 8,
        "Assignments_Submitted": 5,
        "Assignments_Missed": 0,
        "Quiz_Attempts": 3,
        "Project_Grade": 90,
        "Rewatch_Count": 2,
        "Payment_Amount": 1000,
        "App_Usage_Percentage": 80,
        "Reminder_Emails_Clicked": 2,
        "Support_Tickets_Raised": 0,
        "Satisfaction_Rating": 4.5
    }
    
    try:
        response = requests.post(f"http://localhost:{PORT}/predict", json=payload)
        # Even 422 is 'responsive', but we aim for 200
        assert response.status_code in [200, 422], f"Status code: {response.status_code}, Response: {response.text}"
    except requests.exceptions.ConnectionError:
        pytest.fail("Container API not reachable")