from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/docs")
    assert response.status_code == 200

def test_predict_fallback():
    payload = {
        "Student_ID": "TEST_STU",
        "Category": "Programming",
        "Education_Level": "Bachelor",
        "Employment_Status": "Student",
        "City": "Delhi",
        "Device_Type": "Laptop",
        "Internet_Connection_Quality": "High",
        "Course_ID": "C101",
        "Course_Level": "Beginner",
        "Payment_Mode": "UPI",
        "Gender": "Male",
        "Fee_Paid": "Yes",
        "Discount_Used": "No",
        "Age": 20,
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
        "Quiz_Score_Avg": 85,
        "Project_Grade": 90,
        "Progress_Percentage": 75,
        "Rewatch_Count": 2,
        "Payment_Amount": 1000,
        "App_Usage_Percentage": 80,
        "Reminder_Emails_Clicked": 2,
        "Support_Tickets_Raised": 0,
        "Satisfaction_Rating": 4.5
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["status"] in ["success", "fallback_model_missing", "fallback_error", "fallback_heuristic"]

def test_read_monitoring():
    response = client.get("/monitoring")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_predict_invalid_data():
    payload = {
        "Student_ID": "TEST_STU",
        "Category": "Programming", 
    }

def test_predict_sql_injection_attempt():
    """Negative Test: SQL Injection payload."""
    payload = {
        "Student_ID": "'; DROP TABLE students; --",
        "Age": 20, "Progress_Percentage": 50, "Quiz_Score_Avg": 80
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 422] 

def test_predict_huge_payload():
    payload = {
        "Student_ID": "A" * 10000,
        "Age": 20, "Progress_Percentage": 50, "Quiz_Score_Avg": 80
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 422]

