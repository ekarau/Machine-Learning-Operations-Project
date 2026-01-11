from locust import HttpUser, task, between
import random

class PredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # Random payload generation
        payload = {
            "Student_ID": f"STU_{random.randint(1, 1000)}",
            "Category": random.choice(["Programming", "Design", "Marketing"]),
            "Education_Level": random.choice(["Bachelor", "Master", "High School"]),
            "Employment_Status": "Student",
            "City": "New York",
            "Device_Type": "Laptop",
            "Internet_Connection_Quality": "High",
            "Course_ID": "C101",
            "Course_Level": "Beginner",
            "Payment_Mode": "Card",
            "Age": random.randint(18, 60),
            "Progress_Percentage": random.randint(0, 100),
            "Quiz_Score_Avg": random.randint(50, 100),
            # Add other required fields with dummy values to match valid schema
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
            "Satisfaction_Rating": 4.5,
            "Gender": "F",
            "Fee_Paid": "Yes",
            "Discount_Used": "No"
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    response.success()
                elif "fallback" in data["status"]:
                    response.success() # Fallback is technically a success response code-wise
                else:
                    response.failure(f"Status not success: {data.get('status')}")
            else:
                response.failure(f"Status code: {response.status_code}")
