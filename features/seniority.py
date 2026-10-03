def seniority_compatibility(source_years, candidate_years):
    difference = abs(float(source_years) - float(candidate_years))

    if difference <= 2:
        return 1.0
    if difference <= 5:
        return 0.7
    if difference <= 8:
        return 0.4
    return 0.1
