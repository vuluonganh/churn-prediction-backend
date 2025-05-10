import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("\nTesting Health Check:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_prediction():
    """Test the prediction endpoint with sample data"""
    # Sample data that matches your InputData model
    sample_data = {
        "Dependents": "No",
        "tenure": 24,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 1572.0
    }

    print("\nTesting Prediction:")
    print("Input Data:")
    print(json.dumps(sample_data, indent=2))

    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Starting API Tests...")
    test_health()
    test_prediction() 