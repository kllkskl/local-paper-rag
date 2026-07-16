
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz


# --- Papers ---

def exctract(path): #Converts PDFs into text
    text = ""
    pages = fitz.open(path)
    for page in pages:
        text += page.get_text()
    return text


def chunk(text, max_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunks.append(text[start:end])
        start += max_size - overlap
    return chunks


def connect(papers, metadata, reviews):  #Creates chunks with its own metadata and reviews
    corpus = []
    review_lookup = {r['paper_id']: r['reviews'] for r in reviews}

    metadata_indexed = metadata.set_index('id')

    for paper_id, chunks in papers.items():
        paper_review = review_lookup.get(paper_id, '')

        try:
            meta_row = metadata_indexed.loc[paper_id]
        except KeyError:
            print(f"Skipping : {paper_id} not found in metadata")
            continue
        title = meta_row.get('title', 'Unknown')
        author = meta_row.get('author', 'Unknown')
        topic = meta_row.get('topic', 'Unknown')
        try:
            year = int(meta_row['year'])
        except:
            year = None

        for i, c in enumerate(chunks):
            entry = {
                "chunk_id": f"{paper_id}_chunk_{i}",
                "paper_id": paper_id,
                "text": c,
                "title": title,
                "author": author,
                "year": year,
                "topic": topic,
                "reviews": paper_review
                }
            corpus.append(entry)
    return corpus

def main():
    text1 = exctract("data/papers/data_structure.pdf")
    text2 = exctract("data/papers/economic_paper.pdf")
    text3 = exctract("data/papers/framework_paper.pdf")

    papers = {
        "paper1": chunk(text1),
        "paper2": chunk(text2),
        "paper3": chunk(text3),
    }

    metadata = pd.read_csv("data/metadata/papers.csv")

    with open("data/reviews/reviews.json", "r", encoding="utf-8") as f: #Reviews
        reviews = json.load(f)

    corpus = connect(papers, metadata, reviews)
    os.makedirs("data/corpus", exist_ok=True)
    with open("data/corpus/result.json", "w", encoding='utf-8') as f:
        json.dump(corpus, f, indent=4)

    chunks1 = []
    for c in corpus:
        chunks1.append(c['text'])

    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks1, convert_to_numpy=True).astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    os.makedirs("data/saved", exist_ok=True)
    faiss.write_index(index, "data/saved/index.faiss")

    return corpus


if __name__ == "__main__":
    main()
    print("Hello world")



            







