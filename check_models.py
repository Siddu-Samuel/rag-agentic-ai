from google import genai
from src import config

client = genai.Client(api_key=config.GOOGLE_API_KEY)
for m in client.models.list():
    print(m.name)