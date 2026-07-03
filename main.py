import numpy as np
import os
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import json

load_dotenv()
KEY = os.getenv('API')

index = faiss.read_index("data/saved/index.faiss")

with open("data/corpus/result.json", 'r', encoding='utf-8') as f:
    source = json.load(f)

chunks = []
for s in source:
    chunks.append(s['text'])

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def ask(question):
    embedded_question = embedding_model.encode([question], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(embedded_question, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    client = anthropic.Anthropic(api_key=KEY)
    response = client.messages.create(
        model = 'claude-opus-4-6',
        max_tokens = 1024,
        messages = [
            {
                'role': 'user',
                'content': f"Context\n{context}\n\n Question:{question}"
            }
        ]
    )
    answer_text = "".join( block.text for block in response.content if block.type == "text")
    return answer_text

question = "What is CHIA?"
answer = ask(question)
print(answer)


    