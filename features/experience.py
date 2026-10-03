def experience_similarity(source_years, candidate_years):
    difference = abs(float(source_years) - float(candidate_years))
    return max(0.0, min(1.0, 1.0 - (difference / 10.0)))


def experience_balance(source_years, candidate_years):
    difference = abs(float(source_years) - float(candidate_years))
    return max(0.0, min(1.0, 1.0 - (difference / 8.0)))
