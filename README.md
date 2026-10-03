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
python -m matchmaking.api
```

The API listens on `http://127.0.0.1:8000` and accepts JSON `POST` requests:

- `/api/matches/one-to-one`: `{"source_id":"usr_212","candidate_id":"usr_213"}`
- `/api/matches/one-to-many`: `{"source_id":"usr_212","top_k":10}`

The 1:N response contains at most 10 matches. Both responses include feature
values, explanation, and scoring latency in milliseconds. The 1:1 CLI is
`python one_to_one_matchmaking.py`; the 1:N CLI is `python predict.py`.

## Tests and evaluation

Run engine tests with `python -m unittest discover -s tests`. Training reports
NDCG@10 and pairwise ranking accuracy on held-out synthetic labels. These
metrics evaluate agreement with the generated rubric, not real team outcomes.
