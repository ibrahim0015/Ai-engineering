from rank_bm25 import BM25Okapi
from collections import defaultdict
import numpy as np
from pypdf import PdfReader
import chromadb
import json

from evaluation_metrics import precision_at_k, recall_at_k, mrr


# PDF LOADING
file_path = "iso27001.pdf"
reader = PdfReader(file_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text() or ""
    text += page_text + "\n"


# CHUNKING
chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])


# DOCUMENTS + METADATA 
documents_metadata = {}

for chunk_id, chunk in enumerate(chunks):
    documents_metadata[chunk] = {
        "document_name": file_path,
        "chunk_id": chunk_id,
        "source": "pdf"
    }


# CHROMA DB
client = chromadb.PersistentClient(path="./chroma")

collection = client.get_or_create_collection(name="iso27001")

collection.upsert(
    documents=list(documents_metadata.keys()),
    metadatas=list(documents_metadata.values()),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)


#  BM25 INITIALIZATION 
all_results = collection.get(
    include=["documents", "metadatas"]
)

all_docs = all_results["documents"]
all_metadata = all_results["metadatas"]

# Document text -> Metadata
documents_metadata = {
    doc: meta
    for doc, meta in zip(all_docs, all_metadata)
}

tokenized_docs = [
    doc.lower().split()
    for doc in all_docs
]

bm25 = BM25Okapi(tokenized_docs)


# RETRIEVAL FUNCTION
def retrieve_chunks(query, top_k=3):

    # Vector Search
    vector_results = collection.query(
        query_texts=[query],
        n_results=5
    )

    vector_docs = vector_results["documents"][0]

    # BM25 Search
    bm25_scores = bm25.get_scores(query.lower().split())

    bm25_indices = np.argsort(bm25_scores)[::-1][:10]

    bm25_docs = [
        all_docs[i]
        for i in bm25_indices
    ]

    # Reciprocal Rank Fusion
    k = 60

    rrf_scores = defaultdict(float)

    for rank, doc in enumerate(vector_docs, start=1):
        rrf_scores[doc] += 1 / (k + rank)

    for rank, doc in enumerate(bm25_docs, start=1):
        rrf_scores[doc] += 1 / (k + rank)

    # RRF Reranking
    reranked_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Extract Top-K Chunk IDs
    top_chunk_ids = [
        documents_metadata[doc]["chunk_id"]
        for doc, score in reranked_results[:top_k]
    ]

    return top_chunk_ids


# LOAD EVALUATION DATA
with open("iso27001_evaluation_set.json", "r") as f:
    evaluation_data = json.load(f)



# EVALUATION
results = []

all_rankings = []
all_relevant = []

top_k = 5

for item in evaluation_data:

    query = item["query"]
    relevant_chunks = [item["relevant_chunk_id"]]

    retrieved_chunks = retrieve_chunks(query, top_k=top_k)

    # Calculate individual metrics
    precision = precision_at_k(
        retrieved_chunks, relevant_chunks, top_k
    )

    recall = recall_at_k(
        retrieved_chunks, relevant_chunks, top_k
    )

    # Store rankings for overall MRR
    all_rankings.append(retrieved_chunks)
    all_relevant.append(relevant_chunks)

    results.append({
        "query": query,
        "relevant_chunks": relevant_chunks,
        "retrieved_chunks": retrieved_chunks,
        "precision": precision,
        "recall": recall
    })




# OVERALL METRICS

avg_precision = np.mean([
    result["precision"] for result in results
])

avg_recall = np.mean([
    result["recall"] for result in results
])

overall_mrr = mrr(all_rankings, all_relevant)


print("\n========== FINAL RESULTS ==========")

print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("MRR:", overall_mrr)


# SAVE RESULTS

output = {
    "individual_results": results,
    "average_metrics": {
        "precision": float(avg_precision),
        "recall": float(avg_recall),
        "mrr": float(overall_mrr)
    }
}

output_json = "evaluation_results.json"

with open(output_json, "w") as f:
    json.dump(output, f, indent=4)
    
print()
print(f"Detailed results successfully saved to {output_json}.")

