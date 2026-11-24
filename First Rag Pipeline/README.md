# PDF-Based RAG with Hybrid Search

A Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents using hybrid retrieval and Google's Gemini API.

This project explores different retrieval strategies, evaluates retrieval performance, and experiments with chunking configurations to understand how they affect RAG systems.

## Features

- PDF text extraction and chunking using PyPDF.
- Vector search using ChromaDB.
- Keyword-based retrieval using BM25.
- Hybrid search combining vector and BM25 results using Reciprocal Rank Fusion (RRF).
- Context retrieval and answer generation using Gemini.
- Retrieval evaluation using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- Evidence-based evaluation using reference text from the original PDF.
- Experiments with different chunk sizes, overlap settings, and retrieval values of K.
- Automated evaluation with results saved for comparison.

## Project Structure

- `first_rag.py` – Basic RAG implementation.
- `first_rag_eval_pipeline.py` – Hybrid retrieval and evaluation pipeline.
- `final_eval_pipeline.py` – Evidence-based retrieval evaluation and chunking experiments.
- `evaluation_metrics.py` – Precision, Recall, and MRR implementations.
- `iso27001_evaluation_set.json` – Evaluation queries and relevant chunk IDs.
- `evaluation_results.json` – Results from the earlier evaluation pipeline.
- `last_eval_results.json` – Results from the chunking and retrieval experiments.
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
2. Open the relevant Python files.
3. Update the `file_path` variable to point to your PDF:

```python
file_path = "your_document.pdf"
```

4. Run the RAG application:

```bash
python first_rag.py
```

**Note:** The provided evaluation datasets and reference evidence are based on the original ISO 27001 document. If you use a different PDF, create a new evaluation dataset with questions, reference text, and corresponding page numbers for that document.

## Running the Project

Run the basic RAG implementation:

```bash
python first_rag.py
```

Run the hybrid retrieval evaluation pipeline:

```bash
python first_rag_eval_pipeline.py
```

Run the evidence-based evaluation pipeline:

```bash
python final_eval_pipeline.py
```

## Evaluation

The project explores two approaches to retrieval evaluation.

### 1. Retrieval Metrics

The earlier evaluation pipeline uses:

- **Precision@K:** Measures the proportion of retrieved chunks that are relevant.
- **Recall@K:** Measures the proportion of relevant chunks found among the retrieved results.
- **MRR:** Measures how highly the first relevant result is ranked.

### 2. Evidence-Based Evaluation

The final evaluation pipeline uses reference evidence extracted from the original PDF to evaluate retrieval performance.

It works by:

- Locating the reference text on its specified PDF page.
- Tracking the position of the reference evidence within the page.
- Checking how much of that evidence is covered by the retrieved chunks.
- Calculating evidence coverage (Recall@K).
- Calculating MRR based on the rank of the first relevant retrieved chunk.

This approach evaluates retrieval based on the actual coverage of relevant evidence rather than relying only on manually assigned chunk IDs.

### Chunking Experiments

The evaluation pipeline allows experiments with different configurations:

| Parameter | Tested values |
|---|---|
| Chunk size | 500, 800, 1000 |
| Overlap | 0, 100 |
| K | 1, 3, 5 |

The results are saved in `last_eval_results.json`, making it easier to compare different configurations and understand how chunking choices affect retrieval performance.

## Future Improvements

- Experiment with different chunking strategies.
- Compare embedding models.
- Optimize hybrid search parameters.
- Implement cross-encoder reranking.
- Explore answer-generation evaluation.
