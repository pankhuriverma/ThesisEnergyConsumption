import requests

# Replace YOUR_API_KEY with your actual OpenAI API key
api_key = "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"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

data = {
    "model": "gpt-3.5-turbo",  # Specify the model you want to use
    "prompt": "Hello",  # Your input to the model
    "temperature": 0.7,  # Controls randomness
    "max_tokens": 150,  # The maximum length of the model's output
}

response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)

if response.status_code == 200:
    print("API call was successful.")
    print("Response:", response.json())
else:
    print("Failed to call API.")
    print("Status code:", response.status_code)
    print("Response:", response.text)
