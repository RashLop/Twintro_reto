import numpy as np

from features.skills import (
    skill_similarity,
    skill_complementarity,
)
from features.experience import (
    experience_similarity,
    experience_balance,
)
from features.roles import (
    role_similarity,
    role_complementarity,
)
from features.seniority import (
    seniority_compatibility,
)
from features.technology import (
    technology_overlap,
    technology_depth,
)
from features.industry import (
    industry_similarity,
)
from features.interests import interest_similarity

FEATURE_NAMES = (
    "skill_similarity",
    "skill_complementarity",
    "role_similarity",
    "role_complementarity",
    "experience_similarity",
    "experience_balance",
    "seniority_compatibility",
    "technology_overlap",
    "technology_depth",
    "industry_similarity",
    "interest_similarity",
)


def build_features(source, candidate):
    source_profile = source["professional_profile"]
    candidate_profile = candidate["professional_profile"]

    source_skills = source_profile.get("skills", [])
    candidate_skills = candidate_profile.get("skills", [])
    source_role = source_profile.get("role", "")
    candidate_role = candidate_profile.get("role", "")
    source_years = float(source_profile.get("experience_years", 0.0))
    candidate_years = float(candidate_profile.get("experience_years", 0.0))
    source_industry = source_profile.get("industry", "")
    candidate_industry = candidate_profile.get("industry", "")
    source_interests = source_profile.get("interests", [])
    candidate_interests = candidate_profile.get("interests", [])

    features = np.array([
        skill_similarity(source_skills, candidate_skills),
        skill_complementarity(source_skills, candidate_skills),
        role_similarity(source_role, candidate_role),
        role_complementarity(source_role, candidate_role),
        experience_similarity(source_years, candidate_years),
        experience_balance(source_years, candidate_years),
        seniority_compatibility(source_years, candidate_years),
        technology_overlap(source_skills, candidate_skills),
        technology_depth(source_skills, candidate_skills),
        industry_similarity(source_industry, candidate_industry),
        interest_similarity(source_interests, candidate_interests),
    ], dtype=np.float64)

    return features


from matchmaking.explanation import explain_match
