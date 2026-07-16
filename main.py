import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
import json
from openai import OpenAI

index = faiss.read_index("data/saved/index.faiss")

with open("data/corpus/result.json", 'r', encoding='utf-8') as f:
    source = json.load(f)

chunks = []
for s in source:
    chunks.append(s['text'])

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def ask(question):
    embedded_question = embedding_model.encode([question], convert_to_numpy=True).astype('float32')
    _, indices = index.search(embedded_question, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    client = OpenAI(
        base_url = "http://127.0.0.1:8080/v1",
        api_key="not_needed"
    )
    
    response = client.chat.completions.create(
        model = "qwen2.5-7b",
        messages = [
            {"role": "system", "content": "Answer only using provided context"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    print(response.choices[0].message.content)

question = "Where does CHIA help?"
answer = ask(question)
print(answer)


    