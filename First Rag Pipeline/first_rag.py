from google import genai
from google.genai.types import GenerateContentConfig
import os
from rank_bm25 import BM25Okapi
from collections import defaultdict
import numpy as np
from pypdf import PdfReader
import chromadb


# PDF Loading
file_path = "iso27001.pdf"
reader = PdfReader(file_path)

text = ""

for page_num, page in enumerate(reader.pages):
    page_text = page.extract_text() or ""
    text += page_text + "\n"


# Chunking
chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])


# Documents + Metadata Dictionary
documents_metadata = {}

for chunk_id, chunk in enumerate(chunks):
    documents_metadata[chunk] = {
        "document_name": file_path,
        "chunk_id": chunk_id,
        "source": "pdf"
    }


# Chroma DB
client = chromadb.PersistentClient(path="./chroma")

collection = client.get_or_create_collection(name="iso27001")

collection.upsert(
    documents=list(documents_metadata.keys()),
    metadatas=list(documents_metadata.values()),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)


# Search
query = input("Enter your Query: ")


# VECTOR SEARCH 
vector_results = collection.query(
    query_texts=[query],
    n_results=5
)

vector_docs = vector_results["documents"][0]
vector_metadata = vector_results["metadatas"][0]


# BM25 SEARCH 
all_results = collection.get(
    include=["documents", "metadatas"]
)

all_docs = all_results["documents"]
all_metadata = all_results["metadatas"]

# Store retrieved documents and metadata in a dictionary
documents_metadata = {
    doc: meta
    for doc, meta in zip(all_docs, all_metadata)
}

tokenized_docs = [doc.lower().split() for doc in all_docs]

bm25 = BM25Okapi(tokenized_docs)

bm25_scores = bm25.get_scores(query.lower().split())

bm25_indices = np.argsort(bm25_scores)[::-1][:10]

bm25_docs = [all_docs[i] for i in bm25_indices]


# RRF 
k = 60

rrf_scores = defaultdict(float)

for rank, doc in enumerate(vector_docs, start=1):
    rrf_scores[doc] += 1 / (k + rank)

for rank, doc in enumerate(bm25_docs, start=1):
    rrf_scores[doc] += 1 / (k + rank)


# Sort by RRF score
reranked_results = sorted(
    rrf_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


# GENERATE CONTEXT
top_k = 3

context = ""

for doc, score in reranked_results[:top_k]:

    metadata = documents_metadata[doc]

    context += f"""
Document: {metadata['document_name']}
Chunk ID: {metadata['chunk_id']}
Source: {metadata['source']}

Content:
{doc}

"""

# Gemini Client
genai_client = genai.Client(
    api_key=os.getenv("api_key")
)


# System Instructions
with open("system_instructions.md", "r") as f:
    instructions = f.read()


# Generate Answer
response = genai_client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=f"""
Context:
{context}

Question:
{query}
""",
    config=GenerateContentConfig(
        system_instruction=instructions,
    )
)


print("\nAnswer:")
print(response.text)
