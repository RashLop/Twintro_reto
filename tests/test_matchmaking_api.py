import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from matchmaking.api import MatchmakingHandler, _load_user_profiles
from matchmaking.engine import MatchmakingEngine


class FirstFeatureModel:
    def predict(self, features):
        return np.asarray(features)[:, :1]


def make_profile(user_id, name, role, industry, skills):
    return {
        "user_id": user_id,
        "professional_profile": {
            "name": name,
            "role": role,
            "industry": industry,
            "skills": skills,
            "experience_years": 5,
        },
    }


class MatchmakingApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        configured_handler = type(
            "TestMatchmakingHandler",
            (MatchmakingHandler,),
            {
                "engine": MatchmakingEngine(FirstFeatureModel()),
                "profiles": {
                    "usr_101": make_profile(
                        "usr_101", "Alex Rivera", "Backend Developer", "Technology", ["Python", "FastAPI"]
                    ),
                    "usr_202": make_profile(
                        "usr_202", "Sofia Chen", "Data Engineer", "Technology", ["Python", "Spark"]
                    ),
                },
                "user_profiles": {},
                "profiles_path": Path(self.temp_dir.name) / "user_profiles.json",
                "profiles_lock": threading.RLock(),
            },
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), configured_handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp_dir.cleanup()

    def post_json(self, path, payload):
        self.connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def request_json(self, method, path):
        self.connection.request(method, path)
        response = self.connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_web_interface_and_assets_are_served(self):
        for path, expected_type in (
            ("/", "text/html"),
            ("/styles.css", "text/css"),
            ("/app.js", "text/javascript"),
        ):
            with self.subTest(path=path):
                self.connection.request("GET", path)
                response = self.connection.getresponse()
                body = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn(expected_type, response.getheader("Content-Type"))
                self.assertTrue(body)

        self.connection.request("GET", "/api")
        response = self.connection.getresponse()
        api_docs = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 200)
        self.assertEqual(api_docs["service"], "Twintro Matchmaking API")

    def test_profile_options_are_available_for_selectable_form_fields(self):
        status, options = self.request_json("GET", "/api/options")

        self.assertEqual(status, 200)
        self.assertIn("Backend Developer", options["roles"])
        self.assertIn("Technology", options["industries"])
        self.assertIn("Python", options["skills"])
        self.assertIn("Artificial Intelligence", options["interests"])
        self.assertEqual(options["experience_years"], list(range(41)))

    def test_user_can_create_and_list_a_persisted_profile(self):
        profile = make_profile(
            "usr_303", "Jordan Lee", "Product Manager", "Retail", ["Agile"]
        )

        status, created = self.post_json("/api/profiles", profile)
        self.assertEqual(status, 201)
        self.assertEqual(created["profile"], profile)

        status, listing = self.request_json("GET", "/api/profiles")
        self.assertEqual(status, 200)
        self.assertEqual(listing["count"], 3)
        self.assertEqual(
            next(item for item in listing["profiles"] if item["user_id"] == "usr_303")["name"],
            "Jordan Lee",
        )

        with open(Path(self.temp_dir.name) / "user_profiles.json", encoding="utf-8") as file:
            persisted_profiles = json.load(file)
        self.assertEqual(persisted_profiles["usr_303"], profile)
        self.assertEqual(
            _load_user_profiles(Path(self.temp_dir.name) / "user_profiles.json"),
            {"usr_303": profile},
        )

    def test_profile_id_is_generated_when_user_omits_it(self):
        status, response = self.post_json(
            "/api/profiles",
            {
                "professional_profile": {
                    "name": "Taylor Kim",
                    "role": "UX Designer",
                    "industry": "Education",
                    "skills": ["Research"],
                    "experience_years": 2,
                }
            },
        )

        self.assertEqual(status, 201)
        self.assertTrue(response["profile"]["user_id"].startswith("usr_"))
        self.assertIn(response["profile"]["user_id"], self.server.RequestHandlerClass.profiles)

    def test_registered_profiles_can_be_compared_by_id_in_both_modes(self):
        new_profile = make_profile(
            "usr_303", "Jordan Lee", "Product Manager", "Retail", ["Agile"]
        )
        self.assertEqual(self.post_json("/api/profiles", new_profile)[0], 201)

        status, one_to_one = self.post_json(
            "/api/matches/one-to-one",
            {
                "match_mode": "1:1",
                "source_id": "usr_303",
                "candidate_id": "usr_202",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(one_to_one["ranked_matches"][0]["target_user"]["user_id"], "usr_202")

        status, one_to_many = self.post_json(
            "/api/matches/one-to-many",
            {
                "match_mode": "1:N",
                "source_id": "usr_303",
                "candidate_ids": ["usr_101", "usr_202"],
                "top_k": 2,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(one_to_many["ranked_matches"]), 2)
        self.assertEqual(
            {match["target_user"]["user_id"] for match in one_to_many["ranked_matches"]},
            {"usr_101", "usr_202"},
        )

    def test_duplicate_profile_id_is_conflict(self):
        status, response = self.post_json(
            "/api/profiles",
            make_profile("usr_101", "Duplicate", "Role", "Industry", []),
        )

        self.assertEqual(status, 409)
        self.assertIn("Ya existe", response["error"])

    def test_one_to_one_accepts_profiles_and_returns_presentable_match(self):
        status, response = self.post_json(
            "/api/matches/one-to-one",
            {
                "match_mode": "1:1",
                "source_profile": make_profile(
                    "usr_101", "Alex Rivera", "Backend Developer", "Technology", ["Python", "FastAPI"]
                ),
                "candidate_profile": make_profile(
                    "usr_202", "Sofia Chen", "Data Engineer", "Technology", ["Python", "Spark"]
                ),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["match_mode"], "1:1")
        self.assertEqual(response["source_user_id"], "usr_101")
        match = response["ranked_matches"][0]
        self.assertEqual(match["target_user"]["name"], "Sofia Chen")
        self.assertGreaterEqual(match["affinity_percentage"], 0)
        self.assertLessEqual(match["affinity_percentage"], 100)
        self.assertIn("Technology", match["profile_comparison"]["shared_domains"])
        self.assertIsInstance(match["justification"], str)

    def test_one_to_many_accepts_candidate_pool_and_ranks_candidates(self):
        status, response = self.post_json(
            "/api/matches/one-to-many",
            {
                "match_mode": "1:N",
                "source_profile": make_profile(
                    "usr_101", "Alex Rivera", "Backend Developer", "Technology", ["Python", "FastAPI"]
                ),
                "candidate_pool": [
                    make_profile("usr_202", "Sofia Chen", "Data Engineer", "Technology", ["Python", "Spark"]),
                    make_profile("usr_303", "Jordan Lee", "Product Manager", "Retail", ["Agile"]),
                ],
                "top_k": 1,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["match_mode"], "1:N")
        self.assertEqual(len(response["ranked_matches"]), 1)
        self.assertEqual(response["ranked_matches"][0]["target_user"]["user_id"], "usr_202")

    def test_legacy_id_request_keeps_existing_response_fields(self):
        status, response = self.post_json(
            "/api/matches/one-to-one",
            {"source_id": "usr_101", "candidate_id": "usr_202"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["candidate_id"], "usr_202")
        self.assertIn("score", response)
        self.assertIn("explanation", response)

    def test_invalid_profile_is_rejected_with_bad_request(self):
        status, response = self.post_json(
            "/api/matches/one-to-one",
            {
                "source_profile": {"user_id": "usr_101"},
                "candidate_profile": make_profile(
                    "usr_202", "Sofia Chen", "Data Engineer", "Technology", ["Python"]
                ),
            },
        )

        self.assertEqual(status, 400)
        self.assertIn("professional_profile", response["error"])


if __name__ == "__main__":
    unittest.main()
