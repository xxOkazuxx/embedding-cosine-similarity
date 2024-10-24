from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ['API_KEY'])

def get_embeddings(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    
    magnitude1 = sum(a * a for a in vector1) ** 0.5
    magnitude2 = sum(b * b for b in vector2) ** 0.5
    
    # コサイン類似度を計算
    return dot_product / (magnitude1 * magnitude2)


def main():
    vector_1 = get_embeddings(input("入力してください:"))
    vector_2 = get_embeddings(input("入力してください:"))

    print(cosine_similarity(vector_1, vector_2))


if __name__ == "__main__":
    main()