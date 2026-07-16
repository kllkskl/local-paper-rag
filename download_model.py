"""
One-time setup script. Downloads and caches the sentence-transformer 
model locally so preprocess.py and main.py never need network accesss at runtime
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_PATH = "./local_models/all-MiniLM-L6-v2"

def main():
    print(f"Downsloading {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.save(LOCAL_PATH)
    print(f"Saved to {LOCAL_PATH}")

if __name__ == '__main__':
    main()
