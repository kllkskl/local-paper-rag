# Research Paper RAG

A retrieval-augmented question-answering system for research papers, powered
entirely by a local LLM (Qwen2.5-7B-Instruct, Q4 quantization) — no external
API calls, runs on a consumer GPU.

Ask a question about a paper (or a set of papers) and get an answer grounded
in the actual text, in a few seconds.

## Why local?

No API costs, no rate limits, and papers/data never leave your machine —
useful if you're working with unpublished or sensitive material.

## Architecture

1.
**PDF Parsing** - This project uses `PyMuPDF` (`fitz`) for PDF text extraction. 
Note: this is a simpler, faster extraction path than layout-aware parsing 
(e.g. `marker`) — it pulls raw text per page without preserving tables, 
figure structure, or multi-column layout. Sufficient for the current 
text-only embedding pipeline., 
   preserving layout structure and skipping embedded images.
2. **Chunk** — Text is split into chunks, each tagged with metadata
   (paper title, section, etc.) so retrieved chunks stay traceable to source.
3. **Embed & index** — Chunks are embedded with a SentenceTransformer model
   and added to a FAISS index.
4. **Persist** — The FAISS index and corpus are saved to disk after
   preprocessing, so the embedding model doesn't need to reload on every query.
5. **Query** — At query time, the question is embedded the same way, FAISS
   retrieves the top-k relevant chunks, and they're passed as context to the
   local Qwen2.5-7B model to generate the answer.

## Setup

[clone / install / run instructions]

## Example

> **Q:** [a real question you asked]
> **A:** [the model's actual answer]
## Setup

### 1. Clone and install Python dependencies

​```bash
git clone https://github.com/kllkskl/local-paper-rag.git
cd local-paper-rag
pip install -r requirements.txt
​```

### 2. Build llama.cpp with Vulkan support

​```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
​```

This requires the [Vulkan SDK](https://vulkan.lunarg.com/) installed first — the build will fail without it.

### 3. Download the model

Download the Q4_K_M GGUF from [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) and place it in `models/` (or wherever your scripts expect it — match this to your actual config).

### 4. Run preprocessing (builds the FAISS index)

​```bash
python preprocess.py
​```

### 5. Start the local model server

​```bash
cd llama.cpp/build/bin/Release
./llama-server -m path/to/qwen2.5-7b-instruct-q4_k_m.gguf --n-gpu-layers 999 --port 8080
​```

### 6. Ask questions

​```bash
python main.py
​```

## Example

**Question:** "Where does CHIA help?"

**Answer:** CHIA helps in managing and running agentic design patterns, creating CHIA clusters and loops, and providing a runtime environment for user-defined workflows. It also assists in fault tolerance, subprocess tracking, and caching to ensure smooth operation and resource management.