import unittest

import numpy as np

from feature_builder import FEATURE_NAMES, build_features
from matchmaking.engine import MatchmakingEngine


class FirstFeatureModel:
    def __init__(self):
        self.last_batch_size = 0

    def predict(self, features):
        self.last_batch_size = len(features)
        return np.asarray(features)[:, :1]


def make_profile(role, skills, interests=None, years=4, industry="Software"):
    return {
        "professional_profile": {
            "role": role,
            "skills": skills,
            "interests": interests or [],
            "experience_years": years,
            "industry": industry,
        }
    }


class MatchmakingEngineTests(unittest.TestCase):
    def setUp(self):
        self.model = FirstFeatureModel()
        self.engine = MatchmakingEngine(self.model)
        self.source = make_profile("Backend Developer", ["Python", "SQL"], ["AI"])

    def test_one_to_one_uses_model_and_returns_explanation_and_latency(self):
        candidate = make_profile("Data Engineer", ["Python", "Spark"], ["AI"])

        result = self.engine.match_one_to_one(
            self.source, candidate, source_id="source", candidate_id="candidate"
        )

        self.assertGreater(result["score"], 0.0)
        self.assertEqual(result["candidate_id"], "candidate")
        self.assertIn("reasons", result["explanation"])
        self.assertGreaterEqual(result["latency_ms"], 0.0)
        self.assertEqual(len(result["features"]), len(FEATURE_NAMES))

    def test_one_to_many_ranks_top_ten_and_scores_as_a_batch(self):
        candidates = {
            f"candidate-{index}": make_profile(
                "Data Engineer",
                ["Python", f"tool-{index}"] if index % 2 == 0 else [f"tool-{index}"],
            )
            for index in range(12)
        }

        result = self.engine.match_one_to_many(
            self.source, candidates, source_id="source"
        )

        matches = result["matches"]
        self.assertEqual(len(matches), 10)
        self.assertEqual(self.model.last_batch_size, 12)
        self.assertEqual(
            [item["score"] for item in matches],
            sorted((item["score"] for item in matches), reverse=True),
        )
        self.assertGreaterEqual(result["latency_ms"], 0.0)

    def test_interest_similarity_is_part_of_feature_vector(self):
        candidate = make_profile("Backend Developer", ["Python"], ["AI", "Robotics"])

        features = build_features(self.source, candidate)

        self.assertEqual(len(features), 11)
        self.assertGreater(features[FEATURE_NAMES.index("interest_similarity")], 0.0)


if __name__ == "__main__":
    unittest.main()