def normalize_skills(skills):
    if not skills:
        return set()

    normalized = set()
    for skill in skills:
        if not isinstance(skill, str):
            continue
        cleaned = skill.strip().lower()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def skill_similarity(source_skills, candidate_skills):
    source = normalize_skills(source_skills)
    candidate = normalize_skills(candidate_skills)

    if not source and not candidate:
        return 0.0

    union = source | candidate
    if not union:
        return 0.0

    return len(source & candidate) / len(union)


def skill_complementarity(source_skills, candidate_skills):
    source = normalize_skills(source_skills)
    candidate = normalize_skills(candidate_skills)

    if not source or not candidate:
        return 0.0

    shared = source & candidate
    unique = (source | candidate) - shared
    if not unique:
        return 0.0

    return len(unique) / len(source | candidate)
