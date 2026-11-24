import json
import re
import chromadb
from pypdf import PdfReader

file_path = "/content/iso27001.pdf"
reader = PdfReader(file_path)


def normalize(text):
    # Join words broken across lines with a hyphen, e.g. "secu-\nrity" -> "security"
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def create_chunks(chunk_size=500, overlap=0):
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        if not text:
            continue

        step = chunk_size - overlap
        for start in range(0, len(text), step):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": page_num,
                "start": start,
                "end": end,
            })

            if end == len(text):
                break

    return chunks


def build_chroma_collection(
    chunks,
    file_path,
    persist_path="./chroma",
    collection_name="iso27001",
    batch_size=500,
):
    client = chromadb.PersistentClient(path=persist_path)

    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        client.delete_collection(name=collection_name)

    collection = client.get_or_create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    for chunk_id, chunk in enumerate(chunks):
        documents.append(chunk["text"])
        ids.append(f"chunk_{chunk_id}")
        metadatas.append({
            "document_name": file_path,
            "chunk_id": chunk_id,
            "source": "pdf",
            "page": int(chunk["page"]),
            "start": int(chunk["start"]),
            "end": int(chunk["end"]),
        })

    for i in range(0, len(ids), batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

    return collection


def find_evidence(evaluation_item):
    page_num = evaluation_item["page"]
    if not (1 <= page_num <= len(reader.pages)):
        raise ValueError(
            f"Invalid page number {page_num}; PDF has {len(reader.pages)} pages"
        )

    raw_page_text = reader.pages[page_num - 1].extract_text() or ""
    page_text = normalize(raw_page_text)
    evidence = normalize(evaluation_item["relevant_text"])

    if not evidence:
        raise ValueError(f"Empty relevant_text on page {page_num}")

    start = page_text.find(evidence)
    # print(start)
    if start == -1:
        raise ValueError(
            f"Evidence not found on page {page_num}"
    )

    return {
        "page": page_num,
        "start": start,
        "end": start + len(evidence),
    }

def is_relevant(chunk, evidence):
    if chunk["page"] != evidence["page"]:
        return False

    overlap = max(
        0,
        min(chunk["end"], evidence["end"]) - max(chunk["start"], evidence["start"]),
    )
    return overlap > 0


def evaluate(retrieval_function, evaluation_set, k=5):
    if not evaluation_set:
        return {"Recall@K": 0.0, "MRR": 0.0}

    recalls = []
    reciprocal_ranks = []

    for item in evaluation_set:
        try:
            evidence = find_evidence(item)
        except ValueError as e:
            print(f"Skipping item (query={item.get('query')!r}): {e}")
            continue
        results = retrieval_function(item["query"], k)

        intervals = []
        first_relevant_rank = None

        for rank, chunk in enumerate(results, start=1):
            if not is_relevant(chunk, evidence):
                continue

            if first_relevant_rank is None:
                first_relevant_rank = rank

            start = max(chunk["start"], evidence["start"])
            end = min(chunk["end"], evidence["end"])
            intervals.append((start, end))

        intervals.sort()

        covered = 0
        current_start = None
        current_end = None

        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += current_end - current_start
                current_start, current_end = start, end

        if current_start is not None:
            covered += current_end - current_start

        evidence_length = evidence["end"] - evidence["start"]
        recalls.append(covered / evidence_length if evidence_length > 0 else 0.0)
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)

    if not recalls:
        return {"Recall@K": 0.0, "MRR": 0.0}

    return {
        "Recall@K": sum(recalls) / len(recalls),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def retrieval_function(query, k=3):
    if not query or not query.strip():
        return []

    total_docs = collection.count()
    if total_docs == 0 or k <= 0:
        return []

    n_results = min(k, total_docs)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = (results.get("metadatas") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    retrieved_chunks = []
    for rank, (metadata, document, distance) in enumerate(
        zip(metadatas, documents, distances), start=1
    ):
        metadata = metadata or {}
        retrieved_chunks.append({
            "rank": rank,
            "text": document,
            "page": metadata.get("page"),
            "start": metadata.get("start"),
            "end": metadata.get("end"),
            "distance": distance,
        })

    return retrieved_chunks


with open("/content/evaluation_set_without_chunk.json", "r", encoding="utf-8") as f:
    evaluation_set = json.load(f)

configurations = [
    (500, 0),
    (800, 0),
    (1000, 0),
    (500, 100),
    (800, 100),
]
import json

k_values = [1, 3, 5]
all_results = []

for size, overlap in configurations:
    chunks = create_chunks(
        chunk_size=size,
        overlap=overlap
    )

    collection = build_chroma_collection(
        chunks=chunks,
        file_path=file_path,
        persist_path="./chroma",
        collection_name="iso27001",
    )

    for k in k_values:
        scores = evaluate(
            retrieval_function,
            evaluation_set,
            k=k
        )

        result_entry = {
            "k": k,
            "Size": size,
            "Overlap": overlap,
            "Scores": scores
        }

        all_results.append(result_entry)

        print(f"k: {k}, Size: {size}, Overlap: {overlap} -> {scores}")

    

  # 3. Save all results once at the end so previous runs aren't overwritten
with open("last_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
