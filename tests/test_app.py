import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

def test_routes():
    client = app.test_client()

    print("Testing GET / ...")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"VeritasAI" in res.data, "Expected 'VeritasAI' in page"
    print("GET / passed.")

    print("Testing GET /samples ...")
    res = client.get('/samples')
    assert res.status_code == 200
    samples = json.loads(res.data)
    assert len(samples) >= 4
    print(f"GET /samples passed: {len(samples)} samples found.")

    print("Testing POST /predict (Real news text)...")
    payload = {
        "text": "NASA James Webb Space Telescope reveals detailed atmospheric composition of exoplanet orbiting distant star system."
    }
    res = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "success"
    assert data["verdict"] == "REAL"
    print(f"POST /predict Real: Verdict={data['verdict']}, Confidence={data['confidence']}%")

    print("Testing POST /predict (Fake news text)...")
    payload_fake = {
        "text": "SHOCKING PROOF: Government secretly hiding giant alien spacecraft! Miracle cure destroys all cancer in 24 hours!"
    }
    res = client.post('/predict', data=json.dumps(payload_fake), content_type='application/json')
    assert res.status_code == 200
    data_fake = json.loads(res.data)
    assert data_fake["status"] == "success"
    assert data_fake["verdict"] == "FAKE"
    print(f"POST /predict Fake: Verdict={data_fake['verdict']}, Confidence={data_fake['confidence']}%")

    print("Testing POST /api/predict (REST endpoint)...")
    res_api = client.post('/api/predict', json={"text": "The Federal Reserve announced interest rate stabilization measures."})
    assert res_api.status_code == 200
    data_api = json.loads(res_api.data)
    assert data_api["status"] == "success"
    assert data_api["verdict"] == "REAL"
    print(f"POST /api/predict: Verdict={data_api['verdict']}, Probabilities={data_api['probabilities']}")

    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_routes()
