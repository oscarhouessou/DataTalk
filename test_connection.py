
import requests
import os
from dotenv import load_dotenv
import socket

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY non trouvée dans .env")
    exit(1)

print(f"Testing connectivity to api.groq.com...")

# 1. DNS Resolution
try:
    ip = socket.gethostbyname("api.groq.com")
    print(f"DNS Resolution OK: api.groq.com -> {ip}")
except Exception as e:
    print(f"DNS Resolution FAILED: {e}")

# 2. Direct Request
print("\nTesting direct API request...")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5
}

try:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
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
