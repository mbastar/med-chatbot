import requests

print(
    requests.post(
        "https://researchergpt.onrender.com",
        json={
            "query": "what is the weather in Sydney?"
        }
    ).json()
)