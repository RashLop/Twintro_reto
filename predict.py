import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_builder import build_features
from matchmaking.engine import MatchmakingEngine
from neural_network import NeuralNetwork

PROFILES_PATH = ROOT / "data" / "profiles.json"
MODEL_PATH = ROOT / "matchmaking_model.npz"
TOP_K = 10

with open(PROFILES_PATH, "r", encoding="utf-8") as file:
    profiles = json.load(file)

profile_ids = list(profiles.keys())
if len(profile_ids) < 2:
    raise ValueError("Se necesitan al menos 2 perfiles.")

sample_feature = build_features(
    profiles[profile_ids[0]],
    profiles[profile_ids[1]],
)

model = NeuralNetwork(
    input_size=len(sample_feature),
    hidden_sizes=(32, 16, 8),
    learning_rate=0.001,
)
model.load(MODEL_PATH)
engine = MatchmakingEngine(model)

source_id = random.choice(profile_ids)
source = profiles[source_id]
source_profile = source["professional_profile"]

ranking = engine.match_one_to_many(
    source,
    profiles,
    source_id=source_id,
    top_k=TOP_K,
)
top_matches = ranking["matches"]

print()
print("=" * 70)
print("1:N MATCHMAKING - EXPECTED TEAM PERFORMANCE")
print("=" * 70)
print()
print("SOURCE PROFILE")
print("-" * 70)
print(f"ID:          {source_id}")
print(f"Role:        {source_profile['role']}")
print(f"Industry:    {source_profile['industry']}")
print(f"Experience:  {source_profile['experience_years']} años")
print(f"Skills:      {', '.join(source_profile['skills'])}")

print()
print("TOP 10 MATCHES")
print("-" * 70)
print(f"{'Rank':<6}{'Candidate':<20}{'Performance':>15}")
for position, match in enumerate(top_matches, start=1):
    print(f"{position:<6}{match['candidate_id']:<20}{match['score'] * 100:>14.2f}%")
    reasons = " ".join(match["explanation"]["reasons"])
    print(f"      Motivos: {reasons or 'No hay suficientes factores destacados.'}")
    considerations = match["explanation"]["considerations"]
    if considerations:
        print(f"      Diferencias: {' '.join(considerations)}")
print(f"Tiempo de scoring: {ranking['latency_ms']:.2f} ms")
print("=" * 70)
