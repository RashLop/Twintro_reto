import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_builder import FEATURE_NAMES
from matchmaking.engine import MatchmakingEngine
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

    model = NeuralNetwork(
        input_size=len(FEATURE_NAMES),
        hidden_sizes=(32, 16, 8),
        learning_rate=0.001,
    )
    model.load(MODEL_PATH)
    result = MatchmakingEngine(model).match_one_to_one(
        profile_a,
        profile_b,
        source_id=first_id,
        candidate_id=second_id,
    )

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
    for feature_name, feature_value in result["features"].items():
        print(f"{feature_name.replace('_', ' ').title():<28}{feature_value:.4f}")

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Match score: {result['score']:.4f}")
    print(f"Match:       {result['score'] * 100:.2f}%")
    print()
    print("¿POR QUÉ ESTE EMPAREJAMIENTO?")
    for reason in result["explanation"]["reasons"]:
        print(f"- {reason}")
    if not result["explanation"]["reasons"]:
        print("- No hay suficientes factores destacados para explicar el resultado.")
    if result["explanation"]["considerations"]:
        print("Diferencias a considerar:")
        for consideration in result["explanation"]["considerations"]:
            print(f"- {consideration}")
    print(f"Tiempo de scoring: {result['latency_ms']:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
