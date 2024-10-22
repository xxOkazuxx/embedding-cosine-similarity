from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ['API_KEY'])

def get_embeddings(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

print(get_embeddings("こんにちは!ChatGPT!"))