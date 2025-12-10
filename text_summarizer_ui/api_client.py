import requests

API_URL = "http://127.0.0.1:8000"

def summarize(text, mode):
    payload = {"text": text, "mode": mode}
    response = requests.post(f"{API_URL}/summarize", json=payload)
    return response.json().get("summary", "Error")

def compare(text):
    payload = {"text": text}
    response = requests.post(f"{API_URL}/compare", json=payload)
    return response.json()
