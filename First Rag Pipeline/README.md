# PDF-Based RAG with Hybrid Search

A Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents using hybrid retrieval and Google's Gemini API.

## Features

- PDF text extraction and chunking using PyPDF.
- Vector search using ChromaDB.
- Keyword-based retrieval using BM25.
- Hybrid search combining vector and BM25 results using Reciprocal Rank Fusion (RRF).
- Context retrieval and answer generation using Gemini.
- Retrieval evaluation using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).

## Project Structure

- `first_rag.py` – Basic RAG implementation.
- `first_rag_eval_pipeline.py` – Hybrid retrieval and evaluation pipeline.
- `evaluation_metrics.py` – Precision, Recall, and MRR implementations.
- `iso27001_evaluation_set.json` – Evaluation queries and relevant chunk IDs.
- `evaluation_results.json` – Evaluation results.
- `system_instructions.md` – Instructions for the Gemini model.

## Tech Stack

- Python
- Google Gemini API
- ChromaDB
- BM25 (rank-bm25)
- NumPy
- PyPDF

## Setup

Install the required packages:

```bash
pip install google-genai chromadb rank-bm25 numpy pypdf python-dotenv
```

Create a `.env` file in the project directory:

```env
api_key=YOUR_GEMINI_API_KEY
```

Make sure `.env` is included in `.gitignore`.

## Using Your Own PDF

The original ISO 27001 PDF is not included in this repository. You can use your own PDF document instead.

1. Place your PDF file in the project directory.
2. Open `first_rag.py` and `first_rag_eval_pipeline.py`.
3. Update the `file_path` variable to point to your PDF:

```python
file_path = "your_document.pdf"
```

4. Run the RAG application:

```bash
python first_rag.py
```

**Note:** The provided evaluation dataset contains chunk IDs from the original ISO 27001 document. If you use a different PDF, create a new evaluation dataset with relevant chunk IDs corresponding to your document before running the evaluation pipeline.

## Running the Project

Run the basic RAG implementation:

```bash
python first_rag.py
```

Run the retrieval evaluation pipeline:

```bash
python first_rag_eval_pipeline.py
```

## Evaluation

The retrieval pipeline is evaluated using:

- **Precision@K:** Measures how many retrieved chunks are relevant.
- **Recall@K:** Measures how many relevant chunks were retrieved.
- **MRR:** Measures how highly the first relevant chunk is ranked.

Evaluation results are stored in `evaluation_results.json`.

## Future Improvements

- Experiment with different chunking strategies.
- Compare embedding models.
- Optimize hybrid search parameters.
- Implement cross-encoder reranking.