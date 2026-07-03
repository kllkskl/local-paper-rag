import fitz
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# --- Papers ---

def extract(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

text2 = extract("data/papers/economic_paper.pdf")
text1 = extract("data/papers/framework_paper.pdf")


def chunk(text, max_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunks.append(text[start:end])
        start += max_size - overlap
    return chunks

# --- MetaData ---

metadata = pd.read_csv("data/metadata/papers.csv")

# --- JSON file ---

with open("data/reviews/reviews.json", "r", encoding="utf-8") as f:
    reviews = json.load(f)




def connect(papers):
    corpus = []
    review_lookup = {r['paper_id']: r['reviews'] for r in reviews}

    metadata_indexed = metadata.set_index('id')

    for paper_id, chunks in papers.items():
        paper_review = review_lookup.get(paper_id, '')

        try:
            meta_row = metadata_indexed.loc[paper_id]
        except KeyError:
            continue

        for i, chunk in enumerate(chunks):
            entry = {
                "chunk_id": f"{paper_id}_chunk_{i}",
                "paper_id": paper_id,
                "text": chunk,
                "title": meta_row['title'],
                "author": meta_row['author'],
                "year": int(meta_row['year']),
                "topic": meta_row['topic'],
                "reviews": paper_review
                }
            corpus.append(entry)
    return corpus


papers = {
    "paper1": chunk(text1),
    "paper2": chunk(text2)
}


corpus = connect(papers)
print(len(corpus))

os.makedirs("data/corpus", exist_ok=True)
with open("data/corpus/result.json", "w", encoding='utf-8') as f:
    json.dump(corpus, f, indent=4)

with open("data/corpus/result.json", "r", encoding="utf-8") as file:
     source = json.load(file)
chunks1 = []
for s in source:
    chunks1.append(s['text'])

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(chunks1, convert_to_numpy=True).astype('float32')
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss_path = os.makedirs("data/saved", exist_ok=True)
faiss.write_index(index, "data/saved/index.faiss")



            







