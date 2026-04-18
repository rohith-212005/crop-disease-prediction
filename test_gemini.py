from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list_models():
    print(m.name)