from time import perf_counter

import numpy as np

from feature_builder import FEATURE_NAMES, build_features
from matchmaking.explanation import explain_match

MAX_TOP_K = 10


class MatchmakingEngine:
    def __init__(self, model):
        self.model = model

    def match_one_to_one(self, source, candidate, source_id=None, candidate_id=None):
        started = perf_counter()
        feature_vector = build_features(source, candidate)
        prediction = self.model.predict(feature_vector.reshape(1, -1))
        score = float(np.asarray(prediction).reshape(-1)[0])
        result = {
            "source_id": source_id,
            "candidate_id": candidate_id,
            "score": score,
            "features": dict(zip(FEATURE_NAMES, map(float, feature_vector))),
            "explanation": explain_match(source, candidate, feature_vector),
        }
        result["latency_ms"] = (perf_counter() - started) * 1000.0
        return result

    def match_one_to_many(self, source, candidates, source_id=None, top_k=MAX_TOP_K):
        started = perf_counter()
        top_k = int(top_k)
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k debe estar entre 1 y {MAX_TOP_K}.")

        candidate_items = [
            (candidate_id, candidate)
            for candidate_id, candidate in candidates.items()
            if candidate_id != source_id
        ]
        if not candidate_items:
            raise ValueError("No hay candidatos válidos para comparar.")

        feature_vectors = [
            build_features(source, candidate)
            for _, candidate in candidate_items
        ]
        predictions = np.asarray(
            self.model.predict(np.asarray(feature_vectors, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if len(predictions) != len(candidate_items):
            raise ValueError("El modelo devolvió una cantidad inesperada de scores.")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("El modelo devolvió scores no finitos.")

        matches = []
        for (candidate_id, candidate), features, score in zip(
            candidate_items, feature_vectors, predictions
        ):
            matches.append({
                "candidate_id": candidate_id,
                "score": float(score),
                "features": dict(zip(FEATURE_NAMES, map(float, features))),
                "explanation": explain_match(source, candidate, features),
            })
        matches.sort(key=lambda match: match["score"], reverse=True)

        return {
            "source_id": source_id,
            "matches": matches[:top_k],
            "latency_ms": (perf_counter() - started) * 1000.0,
        }