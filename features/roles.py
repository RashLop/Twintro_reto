def normalize_role(role):
    if role is None:
        return "unknown"

    role = str(role).lower().strip()
    if not role:
        return "unknown"

    if "backend" in role:
        return "backend"
    if "frontend" in role or "front" in role or "ui" in role:
        return "frontend"
    if "full stack" in role or "fullstack" in role:
        return "fullstack"
    if "machine learning" in role or "ml engineer" in role:
        return "machine_learning"
    if "data scientist" in role:
        return "data_science"
    if "data engineer" in role:
        return "data_engineer"
    if "devops" in role:
        return "devops"
    if "cloud" in role or "architect" in role:
        return "cloud"
    if "security" in role or "cybersecurity" in role:
        return "security"
    if "designer" in role or "ux" in role:
        return "design"
    if "product" in role:
        return "product"
    if "qa" in role or "tester" in role:
        return "qa"
    return role


def role_similarity(source_role, candidate_role):
    source = normalize_role(source_role)
    candidate = normalize_role(candidate_role)

    if source == "unknown" or candidate == "unknown":
        return 0.0

    if source == candidate:
        return 1.0

    family_pairs = {
        "backend": {"backend", "fullstack", "devops", "cloud"},
        "frontend": {"frontend", "fullstack", "design", "product"},
        "fullstack": {"backend", "frontend", "fullstack", "devops", "product"},
        "machine_learning": {"machine_learning", "data_science", "data_engineer"},
        "data_science": {"machine_learning", "data_science", "data_engineer"},
        "data_engineer": {"machine_learning", "data_science", "data_engineer"},
        "devops": {"backend", "devops", "cloud", "fullstack"},
        "cloud": {"backend", "devops", "cloud"},
        "security": {"security", "backend", "cloud"},
        "design": {"frontend", "design", "product"},
        "product": {"frontend", "design", "product", "fullstack"},
        "qa": {"backend", "frontend", "qa"},
    }

    if candidate in family_pairs.get(source, set()):
        return 0.5

    return 0.0


def role_complementarity(source_role, candidate_role):
    source = normalize_role(source_role)
    candidate = normalize_role(candidate_role)

    if source == "unknown" or candidate == "unknown":
        return 0.0

    complementarity = {
        "backend": {"frontend", "devops", "design", "product", "security", "cloud"},
        "frontend": {"backend", "design", "product", "devops"},
        "fullstack": {"backend", "frontend", "devops", "product"},
        "machine_learning": {"data_science", "data_engineer", "backend", "product"},
        "data_science": {"machine_learning", "backend", "product"},
        "data_engineer": {"machine_learning", "backend", "devops"},
        "devops": {"backend", "frontend", "security", "cloud"},
        "cloud": {"backend", "devops", "security"},
        "security": {"backend", "devops", "cloud"},
        "design": {"backend", "frontend", "product"},
        "product": {"backend", "frontend", "design", "machine_learning"},
        "qa": {"backend", "frontend"},
    }

    if candidate in complementarity.get(source, set()):
        return 1.0

    if source == candidate:
        return 0.2

    return 0.0
