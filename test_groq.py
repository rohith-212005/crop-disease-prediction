import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. Connect with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    # 2. Make the request
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        model="llama-3.3-70b-versatile",
    )
    # Extracting the actual text from the AI's response
    print("GROQ API SUCCESS")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    # 3. Handle errors
    print(f"GROQ API ERROR: {e}")
