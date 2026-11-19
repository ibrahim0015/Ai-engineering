def precision_at_k(retrieved_ids, relevant_ids, k):

    retrieved = retrieved_ids[:k]
    relevant = set(relevant_ids)

    if k == 0:
        return 0.0

    found = sum(
        1 for doc_id in retrieved
        if doc_id in relevant
    )

    return found / k


def recall_at_k(retrieved_ids, relevant_ids, k):

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    found = retrieved.intersection(relevant)

    return len(found) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):

    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in relevant:
            return 1 / rank

    return 0.0

def mrr(all_rankings, all_relevant):

    reciprocal_ranks = []

    for retrieved, relevant in zip(all_rankings, all_relevant):

        rr = reciprocal_rank(retrieved, relevant)
        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
