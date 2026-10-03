import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_builder import build_features
from matchmaking.engine import MatchmakingEngine
from neural_network import NeuralNetwork

PROFILES_PATH = ROOT / "data" / "profiles.json"
MODEL_PATH = ROOT / "matchmaking_model.npz"


def load_engine():
    with PROFILES_PATH.open("r", encoding="utf-8") as file:
        profiles = json.load(file)
    if len(profiles) < 2:
        raise ValueError("Se necesitan al menos dos perfiles.")

    profile_ids = list(profiles)
    input_size = len(build_features(profiles[profile_ids[0]], profiles[profile_ids[1]]))
    model = NeuralNetwork(input_size=input_size)
    model.load(MODEL_PATH)
    return MatchmakingEngine(model), profiles


class MatchmakingHandler(BaseHTTPRequestHandler):
    engine = None
    profiles = None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_json(200, {
                "service": "Twintro Matchmaking API",
                "status": "ok",
                "endpoints": {
                    "one_to_one": {
                        "method": "POST",
                        "path": "/api/matches/one-to-one",
                        "body": {"source_id": "usr_212", "candidate_id": "usr_213"},
                    },
                    "one_to_many": {
                        "method": "POST",
                        "path": "/api/matches/one-to-many",
                        "body": {"source_id": "usr_212", "top_k": 10},
                    },
                },
            })
        elif path == "/health":
            self._send_json(200, {"status": "ok"})
        elif path in ("/api/matches/one-to-one", "/api/matches/one-to-many"):
            self._send_json(405, {"error": "Este endpoint requiere POST.", "allowed_method": "POST"})
        else:
            self._send_json(404, {"error": "Ruta no encontrada."})

    def do_POST(self):
        started = time.perf_counter()
        try:
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("El cuerpo de la solicitud está vacío o es demasiado grande.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("El JSON debe ser un objeto.")

            source_id = payload.get("source_id")
            if source_id not in self.profiles:
                raise KeyError(f"No existe el perfil de origen '{source_id}'.")
            source = self.profiles[source_id]

            if path == "/api/matches/one-to-one":
                candidate_id = payload.get("candidate_id")
                if candidate_id not in self.profiles:
                    raise KeyError(f"No existe el candidato '{candidate_id}'.")
                if candidate_id == source_id:
                    raise ValueError("El perfil de origen no puede ser su propio candidato.")
                response = self.engine.match_one_to_one(
                    source,
                    self.profiles[candidate_id],
                    source_id=source_id,
                    candidate_id=candidate_id,
                )
            elif path == "/api/matches/one-to-many":
                candidate_ids = payload.get("candidate_ids", list(self.profiles))
                if not isinstance(candidate_ids, list):
                    raise ValueError("candidate_ids debe ser una lista de IDs.")
                if any(candidate_id not in self.profiles for candidate_id in candidate_ids):
                    raise KeyError("candidate_ids contiene un perfil inexistente.")
                candidates = {candidate_id: self.profiles[candidate_id] for candidate_id in candidate_ids}
                response = self.engine.match_one_to_many(
                    source,
                    candidates,
                    source_id=source_id,
                    top_k=payload.get("top_k", 10),
                )
            else:
                self._send_json(404, {"error": "Endpoint no encontrado."})
                return

            response["api_latency_ms"] = (time.perf_counter() - started) * 1000.0
            self._send_json(200, response)
        except KeyError as error:
            self._send_json(404, {"error": str(error)})
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            self._send_json(500, {"error": f"Error interno: {error}"})

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


def main():
    engine, profiles = load_engine()
    handler = type(
        "ConfiguredMatchmakingHandler",
        (MatchmakingHandler,),
        {"engine": engine, "profiles": profiles},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 8000), handler)
    print("Matchmaking API disponible en http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()