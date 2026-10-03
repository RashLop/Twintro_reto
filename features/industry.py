def industry_similarity(source_industry, candidate_industry):
    source = str(source_industry or "").strip().lower()
    candidate = str(candidate_industry or "").strip().lower()

    if not source or not candidate:
        return 0.0

    if source == candidate:
        return 1.0

    related = {
        "technology": {"software", "web development", "cloud computing", "artificial intelligence", "technology"},
        "software": {"technology", "software", "web development", "cloud computing", "artificial intelligence"},
        "artificial intelligence": {"technology", "software", "artificial intelligence", "machine learning"},
        "cloud computing": {"technology", "software", "cloud computing", "web development"},
        "web development": {"technology", "software", "web development", "cloud computing", "e-commerce"},
        "e-commerce": {"web development", "software", "e-commerce", "fintech"},
        "fintech": {"e-commerce", "cloud computing", "fintech", "software"},
        "healthtech": {"healthtech", "software", "technology"},
        "edtech": {"edtech", "software", "technology", "education"},
        "gaming": {"gaming", "technology", "software", "design"},
        "design": {"design", "gaming", "web development"},
        "information security": {"information security", "cybersecurity", "technology", "cloud computing"},
        "cybersecurity": {"information security", "cybersecurity", "technology"},
    }

    if candidate in related.get(source, set()):
        return 0.5

    return 0.0
