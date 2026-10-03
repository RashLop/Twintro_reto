def explain_match(source, candidate, features=None):
    if features is None:
        from feature_builder import FEATURE_NAMES, build_features

        features = build_features(source, candidate)
    else:
        from feature_builder import FEATURE_NAMES

    values = dict(zip(FEATURE_NAMES, features))
    source_profile = source.get("professional_profile", {})
    candidate_profile = candidate.get("professional_profile", {})
    source_role = source_profile.get("role", "rol no especificado")
    candidate_role = candidate_profile.get("role", "rol no especificado")
    strengths = []
    considerations = []

    if values["role_complementarity"] >= 0.9:
        strengths.append(f"Los roles de {source_role} y {candidate_role} son complementarios.")
    elif values["role_similarity"] >= 0.5:
        strengths.append(f"Sus roles de {source_role} y {candidate_role} son afines.")
    else:
        considerations.append(f"Los roles de {source_role} y {candidate_role} no tienen una relación directa detectada.")

    source_skills = {
        item.strip().lower(): item.strip()
        for item in source_profile.get("skills", [])
        if isinstance(item, str) and item.strip()
    }
    candidate_skills = {
        item.strip().lower(): item.strip()
        for item in candidate_profile.get("skills", [])
        if isinstance(item, str) and item.strip()
    }
    shared_skills = sorted(source_skills.keys() & candidate_skills.keys())
    source_unique = sorted(source_skills.keys() - candidate_skills.keys())
    candidate_unique = sorted(candidate_skills.keys() - source_skills.keys())

    if shared_skills:
        names = ", ".join(source_skills[item] for item in shared_skills[:3])
        strengths.append(f"Comparten habilidades relevantes: {names}.")
    elif source_unique and candidate_unique and values["skill_complementarity"] > 0:
        source_names = ", ".join(source_skills[item] for item in source_unique[:2])
        candidate_names = ", ".join(candidate_skills[item] for item in candidate_unique[:2])
        strengths.append(f"Aportan habilidades distintas: {source_role} en {source_names}; {candidate_role} en {candidate_names}.")
    else:
        considerations.append("No hay habilidades compartidas registradas.")

    years_a = float(source_profile.get("experience_years", 0.0))
    years_b = float(candidate_profile.get("experience_years", 0.0))
    gap = abs(years_a - years_b)
    if gap <= 2:
        strengths.append("Tienen niveles de experiencia similares.")
    elif gap > 5:
        considerations.append(f"La diferencia de experiencia es de {gap:g} años.")

    if values["technology_overlap"] >= 0.25:
        strengths.append("Comparten tecnologías o herramientas de trabajo.")
    elif values["technology_depth"] >= 0.75:
        strengths.append("Las tecnologías de un perfil están cubiertas por el otro.")

    industry_a = source_profile.get("industry", "")
    industry_b = candidate_profile.get("industry", "")
    if values["industry_similarity"] >= 0.9:
        strengths.append(f"Coinciden en la industria {industry_a}.")
    elif values["industry_similarity"] >= 0.5:
        strengths.append(f"Sus industrias ({industry_a} y {industry_b}) están relacionadas.")

    interests_a = {
        item.strip().lower()
        for item in source_profile.get("interests", [])
        if isinstance(item, str) and item.strip()
    }
    interests_b = {
        item.strip().lower()
        for item in candidate_profile.get("interests", [])
        if isinstance(item, str) and item.strip()
    }
    shared_interests = sorted(interests_a & interests_b)
    if shared_interests:
        strengths.append(f"Comparten intereses: {', '.join(shared_interests[:3])}.")
    elif not interests_a or not interests_b:
        considerations.append("Faltan intereses en uno o ambos perfiles.")

    return {"reasons": strengths[:5], "considerations": considerations[:3]}