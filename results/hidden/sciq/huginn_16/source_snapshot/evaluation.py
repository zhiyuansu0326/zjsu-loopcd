"""Shared multiple-choice input validation and context handling."""
from src.loopcd import Candidate, encode_candidate


def validate_documents(documents, expected_count=None):
    if not documents or (expected_count is not None and len(documents) != expected_count):
        raise ValueError(f"Unexpected document count: {len(documents)}, expected {expected_count}")
    seen = set()
    for doc in documents:
        if not isinstance(doc["id"], str) or doc["id"] in seen:
            raise ValueError("Document IDs must be unique strings.")
        seen.add(doc["id"])
        choices = doc["choices"]
        if len(choices) < 2 or any(not isinstance(c, str) or not c for c in choices):
            raise ValueError(f"{doc['id']}: empty or invalid answer choices")
        if len(doc["labels"]) != len(choices) or len(set(doc["labels"])) != len(choices):
            raise ValueError(f"{doc['id']}: invalid choice labels")
        if type(doc["gold"]) is not int or not 0 <= doc["gold"] < len(choices):
            raise ValueError(f"{doc['id']}: invalid gold index")
        if not isinstance(doc["context"], str) or not doc["context"]:
            raise ValueError(f"{doc['id']}: missing prompt")


def fit_context(candidate, max_length):
    """Match harness [(context + continuation)[-(L+1):]][:-1].

    All answer tokens and their causal prediction positions must survive.
    Return a new candidate and the number of removed context tokens.
    """
    if max_length < 1 or not candidate.targets:
        raise ValueError("A positive model context limit and nonempty target are required.")
    drop = max(0, len(candidate.input_ids) - max_length)
    positions = [p - drop for p in candidate.positions]
    if min(positions) < 0:
        raise ValueError("The answer alone exceeds the context window; refusing to truncate targets.")
    return Candidate(
        candidate.question_index, candidate.choice_index,
        candidate.input_ids[drop:], positions, list(candidate.targets),
    ), drop


def build_candidates(tokenizer, documents, max_length, bos_token_id=None):
    candidates, truncated = [], []
    original_max = 0
    for index, doc in enumerate(documents):
        for choice_index, choice in enumerate(doc["choices"]):
            candidate = encode_candidate(tokenizer, doc["context"], choice, index, choice_index)
            if bos_token_id is not None:
                candidate.input_ids.insert(0, bos_token_id)
                candidate.positions = [p + 1 for p in candidate.positions]
            original_max = max(original_max, len(candidate.input_ids))
            candidate, drop = fit_context(candidate, max_length)
            if drop:
                truncated.append({
                    "question_index": index, "choice_index": choice_index,
                    "removed_context_tokens": drop,
                })
            candidates.append(candidate)
    return candidates, {
        "model_max_length": max_length, "original_max_input_tokens": original_max,
        "candidates_truncated": len(truncated),
        "questions_truncated": len({r["question_index"] for r in truncated}),
        "total_removed_context_tokens": sum(r["removed_context_tokens"] for r in truncated),
        "records": truncated,
    }
