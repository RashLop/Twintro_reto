import json
import math
import os
import sys
import tempfile
import threading
import time
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_builder import build_features
from matchmaking.engine import MatchmakingEngine
from neural_network import NeuralNetwork

PROFILES_PATH = ROOT / "data" / "profiles.json"
USER_PROFILES_PATH = ROOT / "data" / "user_profiles.json"
MODEL_PATH = ROOT / "matchmaking_model.npz"
ONE_TO_ONE_PATH = "/api/matches/one-to-one"
ONE_TO_MANY_PATH = "/api/matches/one-to-many"
PROFILES_API_PATH = "/api/profiles"
OPTIONS_API_PATH = "/api/options"
WEB_PATH = ROOT / "web"
MAX_REQUEST_SIZE = 1_000_000
SUGGESTED_INTERESTS = (
    "Artificial Intelligence",
    "Open Source",
    "Entrepreneurship",
    "Mentoring",
    "Product Development",
    "Education",
    "Sustainability",
    "Design",
    "Research",
    "Gaming",
    "Community Building",
    "Startups",
)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def _profile_id(profile, label):
    if not isinstance(profile, dict):
        raise ValueError(f"{label} debe ser un objeto.")
    profile_id = profile.get("user_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError(f"{label}.user_id es obligatorio y debe ser texto.")
    if not isinstance(profile.get("professional_profile"), dict):
        raise ValueError(f"{label}.professional_profile debe ser un objeto.")
    professional_profile = profile["professional_profile"]
    for field in ("name", "role", "industry"):
        value = professional_profile.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"{label}.professional_profile.{field} debe ser texto.")
    for field in ("skills", "interests"):
        value = professional_profile.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{label}.professional_profile.{field} debe ser una lista de textos.")
    experience = professional_profile.get("experience_years", 0)
    if (
        isinstance(experience, bool)
        or not isinstance(experience, (int, float))
        or not math.isfinite(experience)
        or experience < 0
    ):
        raise ValueError(f"{label}.professional_profile.experience_years debe ser un número no negativo.")
    return profile_id.strip()


def _profile_summary(profile):
    professional_profile = profile["professional_profile"]
    return {
        "user_id": profile["user_id"],
        "name": professional_profile.get("name", ""),
        "role": professional_profile.get("role", ""),
        "industry": professional_profile.get("industry", ""),
    }


def _profile_options(profiles):
    professional_profiles = [
        profile["professional_profile"]
        for profile in profiles.values()
        if isinstance(profile.get("professional_profile"), dict)
    ]

    def values_for(field):
        values = {
            item.strip()
            for profile in professional_profiles
            for item in profile.get(field, [])
            if isinstance(item, str) and item.strip()
        }
        return sorted(values, key=str.casefold)

    def scalar_values_for(field):
        values = {
            profile.get(field, "").strip()
            for profile in professional_profiles
            if isinstance(profile.get(field, ""), str) and profile.get(field, "").strip()
        }
        return sorted(values, key=str.casefold)

    return {
        "roles": scalar_values_for("role"),
        "industries": scalar_values_for("industry"),
        "skills": values_for("skills"),
        "interests": sorted(set(values_for("interests")) | set(SUGGESTED_INTERESTS), key=str.casefold),
        "experience_years": list(range(0, 41)),
    }


def _load_user_profiles(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        profiles = json.load(file)
    if not isinstance(profiles, dict):
        raise ValueError("El registro de perfiles de usuario debe ser un objeto JSON.")
    for profile_id, profile in profiles.items():
        if _profile_id(profile, f"Perfil guardado '{profile_id}'") != profile_id:
            raise ValueError(f"El ID '{profile_id}' no coincide con su perfil guardado.")
    return profiles


def _save_user_profiles(path, profiles):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(profiles, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _profile_comparison(source, target, match):
    source_data = source["professional_profile"]
    target_data = target["professional_profile"]
    source_industry = source_data.get("industry", "")
    target_industry = target_data.get("industry", "")
    shared_domains = []
    if source_industry and target_industry and source_industry.casefold() == target_industry.casefold():
        shared_domains.append(source_industry)

    source_interests = {
        item.strip().casefold(): item.strip()
        for item in source_data.get("interests", [])
        if isinstance(item, str) and item.strip()
    }
    target_interests = {
        item.strip().casefold(): item.strip()
        for item in target_data.get("interests", [])
        if isinstance(item, str) and item.strip()
    }
    shared_domains.extend(
        source_interests[key]
        for key in sorted(source_interests.keys() & target_interests.keys())
    )

    strengths = []
    source_role = source_data.get("role", "")
    target_role = target_data.get("role", "")
    if match["features"].get("role_complementarity", 0.0) >= 0.9 and source_role and target_role:
        strengths.append(f"Roles complementarios: {source_role} + {target_role}")

    source_skills = {
        item.strip().casefold(): item.strip()
        for item in source_data.get("skills", [])
        if isinstance(item, str) and item.strip()
    }
    target_skills = {
        item.strip().casefold(): item.strip()
        for item in target_data.get("skills", [])
        if isinstance(item, str) and item.strip()
    }
    source_unique = sorted(source_skills.keys() - target_skills.keys())
    target_unique = sorted(target_skills.keys() - source_skills.keys())
    if source_unique:
        strengths.append(
            f"Aporta {source_role or 'el perfil de origen'}: "
            + ", ".join(source_skills[key] for key in source_unique[:3])
        )
    if target_unique:
        strengths.append(
            f"Aporta {target_role or 'el perfil candidato'}: "
            + ", ".join(target_skills[key] for key in target_unique[:3])
        )

    explanation = match["explanation"]
    justification_parts = explanation.get("reasons", []) + explanation.get("considerations", [])
    justification = " ".join(justification_parts)
    if not justification:
        justification = "La afinidad se calculó comparando los perfiles profesionales."

    score = float(match["score"])
    if not math.isfinite(score):
        raise ValueError("El modelo devolvió un score no válido.")
    target_user = {
        "user_id": match["candidate_id"],
        "name": target_data.get("name", ""),
    }
    if target_role:
        target_user["role"] = target_role
    if target_industry:
        target_user["industry"] = target_industry

    return {
        "target_user": target_user,
        "affinity_percentage": round(max(0.0, min(1.0, score)) * 100, 2),
        "profile_comparison": {
            "shared_domains": shared_domains,
            "complementary_strengths": strengths,
        },
        "justification": justification,
    }


def _new_schema_response(mode, source, matches, candidates, latency_ms):
    return {
        "match_mode": mode,
        "source_user_id": source["user_id"],
        "ranked_matches": [
            _profile_comparison(source, candidates[match["candidate_id"]], match)
            for match in matches
        ],
        "api_latency_ms": latency_ms,
    }


def _check_match_mode(payload, expected):
    requested_mode = payload.get("match_mode")
    if requested_mode is None:
        return
    accepted_modes = {
        "1:1": {"1:1", "1-1", "1-a-1"},
        "1:N": {"1:n", "1-n", "1-a-n"},
    }
    if not isinstance(requested_mode, str) or requested_mode.strip().casefold() not in accepted_modes[expected]:
        raise ValueError(f"match_mode no coincide con el endpoint; se esperaba '{expected}'.")


def _read_request_payload(handler):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as error:
        raise ValueError("Content-Length no es válido.") from error
    if length <= 0 or length > MAX_REQUEST_SIZE:
        raise ValueError("El cuerpo de la solicitud está vacío o es demasiado grande.")
    try:
        payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("El cuerpo debe contener JSON válido en UTF-8.") from error
    if not isinstance(payload, dict):
        raise ValueError("El JSON debe ser un objeto.")
    return payload


def _resolve_source(payload, profiles):
    if "source_profile" in payload:
        source = payload["source_profile"]
        source_id = _profile_id(source, "source_profile")
        return source_id, source, True

    source_id = payload.get("source_id")
    if not isinstance(source_id, str) or source_id not in profiles:
        raise KeyError(f"No existe el perfil de origen '{source_id}'.")
    return source_id, profiles[source_id], "match_mode" in payload


def _resolve_candidate_ids(candidate_ids, profiles):
    if not isinstance(candidate_ids, list):
        raise ValueError("candidate_ids debe ser una lista de IDs.")
    if any(not isinstance(candidate_id, str) for candidate_id in candidate_ids):
        raise ValueError("candidate_ids solo puede contener IDs de texto.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids no puede contener IDs duplicados.")
    missing_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in profiles]
    if missing_ids:
        raise KeyError(f"No existe el perfil '{missing_ids[0]}'.")
    return {candidate_id: profiles[candidate_id] for candidate_id in candidate_ids}


class ProfileConflictError(Exception):
    pass


def load_engine():
    with PROFILES_PATH.open("r", encoding="utf-8") as file:
        profiles = json.load(file)
    user_profiles = _load_user_profiles(USER_PROFILES_PATH)
    conflicts = profiles.keys() & user_profiles.keys()
    if conflicts:
        raise ValueError(
            "El registro de perfiles de usuario contiene IDs ya existentes: "
            + ", ".join(sorted(conflicts))
        )
    profiles.update(user_profiles)
    if len(profiles) < 2:
        raise ValueError("Se necesitan al menos dos perfiles.")

    profile_ids = list(profiles)
    input_size = len(build_features(profiles[profile_ids[0]], profiles[profile_ids[1]]))
    model = NeuralNetwork(input_size=input_size)
    model.load(MODEL_PATH)
    return MatchmakingEngine(model), profiles, user_profiles


class MatchmakingHandler(BaseHTTPRequestHandler):
    engine = None
    profiles = None
    user_profiles = None
    profiles_path = USER_PROFILES_PATH
    profiles_lock = threading.RLock()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            try:
                body = (WEB_PATH / filename).read_bytes()
            except OSError:
                self._send_json(500, {"error": "No se pudo cargar la interfaz web."})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api":
            self._send_json(200, {
                "service": "Twintro Matchmaking API",
                "status": "ok",
                "endpoints": {
                    "one_to_one": {
                        "method": "POST",
                        "path": ONE_TO_ONE_PATH,
                        "input": {
                            "match_mode": "1:1",
                            "source_profile": {
                                "user_id": "usr_101",
                                "professional_profile": {
                                    "name": "Alex Rivera",
                                    "role": "Senior AI Engineer",
                                    "industry": "Technology",
                                    "skills": ["Python", "FastAPI"],
                                    "experience_years": 8,
                                },
                            },
                            "candidate_profile": {
                                "user_id": "usr_202",
                                "professional_profile": {
                                    "name": "Sofia Chen",
                                    "role": "Backend Developer",
                                    "industry": "Technology",
                                    "skills": ["Python", "Product Strategy"],
                                    "experience_years": 6,
                                },
                            },
                        },
                        "output": {
                            "match_mode": "1:1",
                            "source_user_id": "usr_101",
                            "ranked_matches": [{
                                "target_user": {"user_id": "usr_202", "name": "Sofia Chen"},
                                "affinity_percentage": 98.0,
                                "profile_comparison": {
                                    "shared_domains": ["Technology"],
                                    "complementary_strengths": ["Aporta Backend Developer: Product Strategy"],
                                },
                                "justification": "Perfiles con habilidades y experiencia complementarias.",
                            }],
                        },
                    },
                    "one_to_many": {
                        "method": "POST",
                        "path": ONE_TO_MANY_PATH,
                        "input": {
                            "match_mode": "1:N",
                            "source_profile": {"user_id": "usr_101", "professional_profile": {}},
                            "candidate_pool": [
                                {"user_id": "usr_202", "professional_profile": {}},
                            ],
                            "top_k": 10,
                        },
                        "output": {
                            "match_mode": "1:N",
                            "source_user_id": "usr_101",
                            "ranked_matches": [{
                                "target_user": {"user_id": "usr_202", "name": "Sofia Chen"},
                                "affinity_percentage": 98.0,
                                "profile_comparison": {
                                    "shared_domains": ["Technology"],
                                    "complementary_strengths": ["Aporta Backend Developer: Product Strategy"],
                                },
                                "justification": "Perfiles con habilidades y experiencia complementarias.",
                            }],
                            "api_latency_ms": 2.4,
                        },
                    },
                    "profiles": {
                        "list": {"method": "GET", "path": PROFILES_API_PATH},
                        "create": {
                            "method": "POST",
                            "path": PROFILES_API_PATH,
                            "input": {
                                "professional_profile": {
                                    "name": "Nombre",
                                    "role": "Rol profesional",
                                    "industry": "Industria",
                                    "skills": ["Habilidad"],
                                    "experience_years": 3,
                                },
                            },
                            "profile_options": {"method": "GET", "path": OPTIONS_API_PATH},
                        },
                    },
                },
                "legacy_input": {
                    "one_to_one": {"source_id": "usr_212", "candidate_id": "usr_213"},
                    "one_to_many": {"source_id": "usr_212", "top_k": 10},
                },
            })
        elif path == "/health":
            self._send_json(200, {"service": "Twintro Matchmaking API", "status": "ok"})
        elif path == PROFILES_API_PATH:
            with self.profiles_lock:
                profiles = [
                    _profile_summary(profile)
                    for profile in self.profiles.values()
                ]
            self._send_json(200, {"count": len(profiles), "profiles": profiles})
        elif path == OPTIONS_API_PATH:
            with self.profiles_lock:
                options = _profile_options(self.profiles)
            self._send_json(200, options)
        elif path.startswith(f"{PROFILES_API_PATH}/"):
            profile_id = unquote(path[len(PROFILES_API_PATH) + 1:])
            profile = self.profiles.get(profile_id)
            if profile is None:
                self._send_json(404, {"error": f"No existe el perfil '{profile_id}'."})
            else:
                self._send_json(200, {"profile": profile})
        elif path in (ONE_TO_ONE_PATH, ONE_TO_MANY_PATH):
            self._send_json(405, {"error": "Este endpoint requiere POST.", "allowed_method": "POST"})
        else:
            self._send_json(404, {"error": "Ruta no encontrada."})

    def do_POST(self):
        started = time.perf_counter()
        try:
            path = urlsplit(self.path).path
            if path == PROFILES_API_PATH:
                payload = _read_request_payload(self)
                profile = dict(payload)
                if "user_id" not in profile:
                    profile["user_id"] = f"usr_{uuid4().hex}"
                profile_id = _profile_id(profile, "profile")
                if profile["user_id"] != profile_id:
                    raise ValueError("profile.user_id no puede tener espacios al inicio o al final.")
                with self.profiles_lock:
                    if profile_id in self.profiles:
                        raise ProfileConflictError(f"Ya existe un perfil con ID '{profile_id}'.")
                    updated_user_profiles = dict(self.user_profiles)
                    updated_user_profiles[profile_id] = profile
                    _save_user_profiles(self.profiles_path, updated_user_profiles)
                    self.user_profiles[profile_id] = profile
                    self.profiles[profile_id] = profile
                self._send_json(201, {"profile": profile})
                return
            if path not in (ONE_TO_ONE_PATH, ONE_TO_MANY_PATH):
                self._send_json(404, {"error": "Endpoint no encontrado."})
                return
            payload = _read_request_payload(self)

            source_id, source, uses_new_schema = _resolve_source(payload, self.profiles)

            if path == ONE_TO_ONE_PATH:
                _check_match_mode(payload, "1:1")
                if uses_new_schema:
                    if "candidate_profile" in payload:
                        candidate = payload["candidate_profile"]
                        candidate_id = _profile_id(candidate, "candidate_profile")
                    else:
                        candidate_id = payload.get("candidate_id")
                        if not isinstance(candidate_id, str) or candidate_id not in self.profiles:
                            raise KeyError(f"No existe el candidato '{candidate_id}'.")
                        candidate = self.profiles[candidate_id]
                    if candidate_id == source_id:
                        raise ValueError("El perfil de origen no puede ser su propio candidato.")
                    result = self.engine.match_one_to_one(
                        source,
                        candidate,
                        source_id=source_id,
                        candidate_id=candidate_id,
                    )
                    response = _new_schema_response(
                        "1:1",
                        source,
                        [result],
                        {candidate_id: candidate},
                        (time.perf_counter() - started) * 1000.0,
                    )
                    self._send_json(200, response)
                    return

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
            else:
                _check_match_mode(payload, "1:N")
                if uses_new_schema:
                    if "candidate_pool" in payload:
                        candidate_pool = payload["candidate_pool"]
                        if not isinstance(candidate_pool, list):
                            raise ValueError("candidate_pool debe ser una lista de perfiles.")
                        candidates = {}
                        for index, candidate in enumerate(candidate_pool):
                            candidate_id = _profile_id(candidate, f"candidate_pool[{index}]")
                            if candidate_id in candidates:
                                raise ValueError(f"candidate_pool contiene el ID duplicado '{candidate_id}'.")
                            candidates[candidate_id] = candidate
                    elif "candidate_ids" in payload:
                        candidates = _resolve_candidate_ids(payload["candidate_ids"], self.profiles)
                    else:
                        candidates = dict(self.profiles)
                    result = self.engine.match_one_to_many(
                        source,
                        candidates,
                        source_id=source_id,
                        top_k=payload.get("top_k", 10),
                    )
                    response = _new_schema_response(
                        "1:N",
                        source,
                        result["matches"],
                        candidates,
                        (time.perf_counter() - started) * 1000.0,
                    )
                    self._send_json(200, response)
                    return

                candidate_ids = payload.get("candidate_ids", list(self.profiles))
                candidates = _resolve_candidate_ids(candidate_ids, self.profiles)
                response = self.engine.match_one_to_many(
                    source,
                    candidates,
                    source_id=source_id,
                    top_k=payload.get("top_k", 10),
                )

            response["api_latency_ms"] = (time.perf_counter() - started) * 1000.0
            self._send_json(200, response)
        except ProfileConflictError as error:
            self._send_json(409, {"error": str(error)})
        except KeyError as error:
            self._send_json(404, {"error": str(error)})
        except (ValueError, TypeError) as error:
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
    engine, profiles, user_profiles = load_engine()
    handler = type(
        "ConfiguredMatchmakingHandler",
        (MatchmakingHandler,),
        {
            "engine": engine,
            "profiles": profiles,
            "user_profiles": user_profiles,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 8000), handler)
    print("Matchmaking API disponible en http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()