
import requests
import os
from dotenv import load_dotenv
import socket

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"Testing connectivity to api.openai.com...")

# 1. DNS Resolution
try:
    ip = socket.gethostbyname("api.openai.com")
    print(f"DNS Resolution OK: api.openai.com -> {ip}")
except Exception as e:
    print(f"DNS Resolution FAILED: {e}")

# 2. Direct Request
print("\nTesting direct API request...")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5
}

try:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=10
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! API is working.")
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"Failed with status {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Request FAILED: {e}")
