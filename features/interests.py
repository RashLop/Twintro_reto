def interest_similarity(source_interests, candidate_interests):
    source = {
        interest.strip().lower()
        for interest in source_interests or []
        if isinstance(interest, str) and interest.strip()
    }
    candidate = {
        interest.strip().lower()
        for interest in candidate_interests or []
        if isinstance(interest, str) and interest.strip()
    }
    union = source | candidate
    if not union:
        return 0.0
    return len(source & candidate) / len(union)