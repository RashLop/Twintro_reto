import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_builder import build_features
from neural_network import NeuralNetwork

PROFILES_PATH = ROOT / "data" / "profiles.json"
MODEL_PATH = ROOT / "matchmaking_model.npz"


def load_profiles():
    with open(PROFILES_PATH, "r", encoding="utf-8") as file:
        profiles = json.load(file)

    valid_profiles = {}
    for user_id, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        professional = profile.get("professional_profile")
        if not isinstance(professional, dict):
            continue
        if not professional.get("role") or not professional.get("industry"):
            continue
        if not isinstance(professional.get("skills", []), list):
            continue
        if "experience_years" not in professional:
            continue
        valid_profiles[user_id] = profile

    if len(valid_profiles) < 2:
        raise ValueError("Se necesitan al menos dos perfiles válidos para comparar.")

    return valid_profiles


def print_candidates(profiles):
    print()
    print("=" * 70)
    print("PERFILES DISPONIBLES")
    print("=" * 70)
    for user_id, profile in profiles.items():
        professional = profile["professional_profile"]
        print(
            f"{user_id:<12} | "
            f"{professional['role']:<30} | "
            f"{professional['industry']}"
        )
    print("=" * 70)
    print()


def main():
    profiles = load_profiles()
    print_candidates(profiles)

    first_id = input("Introduce el ID del primer candidato: ").strip()
    second_id = input("Introduce el ID del segundo candidato: ").strip()

    if first_id not in profiles:
        raise ValueError(f"El perfil '{first_id}' no existe en profiles.json.")
    if second_id not in profiles:
        raise ValueError(f"El perfil '{second_id}' no existe en profiles.json.")
    if first_id == second_id:
        raise ValueError("Debes seleccionar dos perfiles diferentes.")

    profile_a = profiles[first_id]
    profile_b = profiles[second_id]

    features = build_features(profile_a, profile_b)
    X = np.asarray([features], dtype=np.float64)

    model = NeuralNetwork(
        input_size=len(features),
        hidden_sizes=(32, 16, 8),
        learning_rate=0.001,
    )
    model.load(MODEL_PATH)

    prediction = model.predict(X)
    score = float(prediction[0, 0])

    a = profile_a["professional_profile"]
    b = profile_b["professional_profile"]

    print()
    print("=" * 70)
    print("MATCHMAKING 1:1")
    print("=" * 70)
    print()
    print("CANDIDATO A")
    print("-" * 70)
    print(f"ID:          {first_id}")
    print(f"Nombre:      {profile_a.get('user_id', 'N/A')}")
    print(f"Rol:         {a['role']}")
    print(f"Industria:   {a['industry']}")
    print(f"Experiencia: {a['experience_years']} años")
    print(f"Skills:      {', '.join(a['skills'])}")

    print()
    print("CANDIDATO B")
    print("-" * 70)
    print(f"ID:          {second_id}")
    print(f"Nombre:      {profile_b.get('user_id', 'N/A')}")
    print(f"Rol:         {b['role']}")
    print(f"Industria:   {b['industry']}")
    print(f"Experiencia: {b['experience_years']} años")
    print(f"Skills:      {', '.join(b['skills'])}")

    print()
    print("FEATURES")
    print("-" * 70)
    print(f"Skill similarity:        {features[0]:.4f}")
    print(f"Skill complementarity:   {features[1]:.4f}")
    print(f"Role similarity:         {features[2]:.4f}")
    print(f"Role complementarity:    {features[3]:.4f}")
    print(f"Experience similarity:   {features[4]:.4f}")
    print(f"Experience balance:       {features[5]:.4f}")
    print(f"Seniority compatibility: {features[6]:.4f}")
    print(f"Technology overlap:      {features[7]:.4f}")
    print(f"Technology depth:        {features[8]:.4f}")
    print(f"Industry similarity:     {features[9]:.4f}")

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Match score: {score:.4f}")
    print(f"Match:       {score * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
