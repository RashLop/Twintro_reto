# Twintro Matchmaking

The existing NumPy neural network is used as the scoring model. It scores
professional affinity and complementarity from role, skills, experience,
technology, industry, and optional profile interests.

## Profile interests

Profiles may include an `interests` string list inside `professional_profile`:

```json
"interests": ["artificial intelligence", "open source"]
```

The current profile data has no interests, so add real interest data before
training if that signal should affect scores. Missing interests contribute
zero to that feature.

## Train and run

After updating profile data, regenerate the synthetic training pairs and train
the model. The feature vector now has 11 inputs, so the existing 10-input model
must be retrained before the API or prediction scripts can load it.

```powershell
python data/training_datagenerator.py
python train.py
```

## Run the API

From the project root, start the API with:

```powershell
python -m matchmaking.api
```

Open `http://127.0.0.1:8000` in a browser to use the web interface: create a
profile, select a source and candidate profile, and view matching results.
The API listens on the same address and continues to accept JSON `POST` requests:

- `GET /`: web interface.
- `GET /api`: service status, endpoint descriptions, and input/output examples.
- `GET /health`: service health.
- `GET /api/profiles`: list available profiles (user ID, name, role, and industry).
- `GET /api/profiles/{user_id}`: get one complete profile.
- `GET /api/options`: get the available roles, industries, skills, interests, and experience values.
- `POST /api/profiles`: register a new profile.
- `POST /api/matches/one-to-one`: compare a source profile with one candidate.
- `POST /api/matches/one-to-many`: rank a source profile against a candidate pool.

Create a profile by sending its user-entered data to `POST /api/profiles`. The
API generates a unique `user_id` and returns the saved profile; clients may
also provide their own unique ID:

```json
{
  "professional_profile": {
    "name": "Alex Rivera",
    "role": "Senior AI Engineer",
    "industry": "Technology",
    "skills": ["Python", "FastAPI"],
    "experience_years": 8,
    "interests": ["AI", "Open Source"]
  }
}
```

User-created profiles are saved in `data/user_profiles.json`, so they remain
available after restarting the API. Their IDs must be unique across the
catalog. Use `GET /api/profiles` to populate a profile selector in a client.
The interface uses predefined role and industry dropdowns, a year selector,
and searchable multi-select skill and interest options to reduce typing errors.

Compare two saved profiles 1:1:

```json
{
  "match_mode": "1:1",
  "source_id": "usr_new",
  "candidate_id": "usr_212"
}
```

Rank a saved profile against selected profiles with 1:N:

```json
{
  "match_mode": "1:N",
  "source_id": "usr_new",
  "candidate_ids": ["usr_212", "usr_213", "usr_214"],
  "top_k": 10
}
```

Omit `candidate_ids` to compare against every available profile except the
source profile. Both matching endpoints also accept unsaved profiles directly
in the request body when a client does not need to register them first. Each
profile uses a `user_id` and a `professional_profile` object:

```json
{
  "match_mode": "1:N",
  "source_profile": {
    "user_id": "usr_101",
    "professional_profile": {
      "name": "Alex Rivera",
      "role": "Senior AI Engineer",
      "industry": "Technology",
      "skills": ["Python", "FastAPI"],
      "experience_years": 8
    }
  },
  "candidate_pool": [
    {
      "user_id": "usr_202",
      "professional_profile": {
        "name": "Sofia Chen",
        "role": "Backend Developer",
        "industry": "Technology",
        "skills": ["Python", "Product Strategy"],
        "experience_years": 6
      }
    }
  ],
  "top_k": 10
}
```

For a direct 1:1 comparison, pass one `candidate_profile` instead of
`candidate_pool`. The 1:N response is sorted by affinity and contains up to 10
results by default. Each ranked result provides a `target_user`,
`affinity_percentage` (0–100), shared domains, complementary strengths, and a
human-readable justification. Responses also include API latency in
milliseconds.

The original ID-based request format remains supported:

```json
{"source_id": "usr_212", "candidate_id": "usr_213"}
{"source_id": "usr_212", "top_k": 10}
```

ID-based responses keep their original score, feature, explanation, and
latency fields. The 1:1 CLI is `python one_to_one_matchmaking.py`; the 1:N CLI
is `python predict.py`.

## Tests and evaluation

Run engine tests with `python -m unittest discover -s tests`. Training reports
NDCG@10 and pairwise ranking accuracy on held-out synthetic labels. These
metrics evaluate agreement with the generated rubric, not real team outcomes.
