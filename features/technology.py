def _normalize_technology_skills(skills):
    normalized = set()
    for skill in skills or []:
        if not isinstance(skill, str):
            continue
        cleaned = skill.strip().lower()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def technology_overlap(source_skills, candidate_skills):
    source = _normalize_technology_skills(source_skills)
    candidate = _normalize_technology_skills(candidate_skills)

    if not source or not candidate:
        return 0.0

    overlap = len(source & candidate)
    union = len(source | candidate)
    if union == 0:
        return 0.0
    return overlap / union


def technology_depth(source_skills, candidate_skills):
    source = _normalize_technology_skills(source_skills)
    candidate = _normalize_technology_skills(candidate_skills)

    if not source or not candidate:
        return 0.0

    smaller_profile_size = min(len(source), len(candidate))
    return len(source & candidate) / smaller_profile_size
