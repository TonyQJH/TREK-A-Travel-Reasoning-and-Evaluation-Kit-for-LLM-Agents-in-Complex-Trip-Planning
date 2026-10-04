# TREK: A Travel Reasoning and Evaluation Kit for LLM Agents in Complex Trip Planning

TREK is a benchmark for **feasible itinerary synthesis** — producing a *single* travel plan that is
jointly (a) constraint-correct, (b) hallucination-free, (c) spatio-temporally executable, (d)
budget-valid, and (e) responsive to the facility requirements operationalized for stated traveler
personas, all at once.

What makes TREK different from prior travel benchmarks:

- **A fully deterministic, rule-based evaluator — no LLM judge.** Every score is an exact computation
  against a versioned knowledge base, so results are bit-reproducible and free to re-run.
- **Gold references with an attainable ceiling under TREK's rules.** The 533 feasible tasks ship
  with human-validated reference itineraries, and the 267 infeasible tasks with typed refusal
  references. All 800 references attain **1.0** on applicable correctness dimensions under the
  published evaluator. You can verify this in one command (below).
- **Typed infeasibility.** 267 of the 800 tasks are *provably* infeasible with a machine-checkable
  cause (route / entity / budget); a correct agent must refuse **and name the right reason**.
- **A production-style tool sandbox.** Agents act through validated RESTful search endpoints with
  structured errors, not free-form database lookups.

The benchmark comprises **800 tasks** (533 feasible, 267 infeasible) over a fixed, source-informed
sandbox knowledge base with synthetic identifiers and calibrated structural fields:
**212,530 records** across **375 cities** and **13 personas**.

---

## Repository layout

```
.
├── verify_gold.py          # one-command proof that the gold reaches 1.0 (offline, no keys)
├── score_trek.py           # the deterministic evaluator (CLI)
├── test_vehicle_capacity.py # offline regression tests for KB-backed capacity checks
├── run_trek.py             # the agent runner (Amazon Bedrock function-calling agent)
├── scoring.py              # the 9-dimension scorer (imported by score_trek.py)
├── implicit_scoring.py     # D1 implicit-need scorer (deterministic set-intersection)
├── travel_time.py          # the door-to-door travel-time model shared by B3 and the agent tool
├── data_loader.py          # loads the knowledge base + task metadata
├── cost_model.py           # the shared cost model (used by the generator, sandbox, and scorer)
├── generate_trek_queries.py# task generator (labels correct by construction)
├── build_v2_kb.py          # knowledge-base build script (provenance/reproducibility)
├── trek_queries.csv        # the 800 tasks
├── trek_gold.jsonl         # the 800 gold references (one per task)
├── trek_models.json        # the 15 evaluated models
├── trek_agent/             # the Bedrock-native, function-calling agent (7 tools)
├── api/                    # the tool sandbox (Flask) + the knowledge base under api/data/v2/
└── results/                # our reported leaderboard and score summary
```

---

## Installation

```bash
# Download this repository and enter it, then:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core: evaluator + agent runner
pip install -r api/requirements.txt      # extra: the tool sandbox (faiss, torch, ...)
```

Python 3.9+ is recommended. The evaluator (`verify_gold.py`, `score_trek.py`) needs only
`requirements.txt`; the sandbox and live agent runs need the extras.

---

## Quickstart

### 1. Verify the achievable ceiling (offline, no API keys)

This scores the gold references and confirms every task reaches **1.0** — the paper's central claim,
reproducible on any machine with no network:

```bash
python verify_gold.py
```

Expected output:

```
gold task-perfect rate:  feasible = 1.0   infeasible = 1.0
PASS — the ceiling is demonstrably reachable: every gold task scores 1.0.
```

(It re-scores all 800 tasks and takes ~1–2 minutes on first run while the knowledge base loads.)

### 2. Score your own agent's outputs

Put one JSONL file per model under a directory, named `trek_<label>.jsonl`, with one result per line
(`{"is_feasible": ..., "plan": {...}, "refusal_reason": ...}`, plus `query_index` and `submitted`;
see `run_trek.py` for the exact schema). Then:

```bash
python score_trek.py --results-dir path/to/your/results --out-dir path/to/scores
```

It writes a per-task `<label>.scores.csv`, a `summary.json`/`summary.csv` leaderboard, and prints the
headline **task-perfect rate** and category scores for each model. Missing or crashed tasks are scored
as failed submissions, so the denominator is identical for every model.

### 3. Run the tool sandbox

```bash
TREK_API_PORT=5001 python api/app.py
```

This serves the four search endpoints (flights, hotels, attractions, cars) plus `submit_plan` over the
knowledge base. Semantic `amenity`/`facility` search uses embeddings (Amazon Bedrock Titan by default;
see `api/bedrock_embed.py`, or the local Sentence-Transformers option in `api/embedding.py`).

### 4. Run an agent end-to-end (needs Amazon Bedrock access)

Provide AWS credentials (see **Credentials** below), start the sandbox, then:

```bash
# a quick 2-task smoke test with one model
python run_trek.py --models moonshotai.kimi-k2.5 --limit 2 \
                   --api-url http://localhost:5001

# the full 15-model run reported in the paper
python run_trek.py --models-file trek_models.json --api-url http://localhost:5001 --concurrency 16
python score_trek.py --results-dir trek_results --out-dir trek_scores
```

`run_trek.py` is resumable (it skips any `query_index` already in the output) and writes one
`trek_<label>.jsonl` per model under `--results-dir` (default `trek_results/`).

---

## The evaluator

Every submission is scored on **nine correctness dimensions** in four categories, plus a separate
efficiency (resource-usage) axis:

| Category | Dimensions | Question |
|---|---|---|
| Constraint Satisfaction | D0-key, D1, D2, D3 | Are the requested entities, implicit needs, cities, and budget met? |
| Truthfulness | D0-src | Does every named entity resolve to a real KB record? (zero-tolerance) |
| Executability | B2, B3 | Are visits within opening hours and days physically traversable? |
| Infeasibility Handling | D4 | On infeasible tasks, is the refusal correct **and** the right typed cause? |

The four categories combine with a **geometric mean**, and the headline metric is the **task-perfect
rate**: the fraction of tasks solved on *every* applicable dimension (reported separately over the 533
feasible and 267 infeasible tasks). D1 (implicit needs) is scored by deterministic facility
set-intersection — no embeddings, no LLM. Full definitions are in the paper.

When a task requires rental cars, D0-key also requires a car in each requested stay city with
KB-backed passenger capacity at least the larger of the party size and requested capacity. An
agent's claimed capacity is not accepted as evidence. This check remains within D0-key's existing
all-or-nothing conjunction; it adds no new dimension or validity gate. Run its offline tests with:

```bash
python test_vehicle_capacity.py -v
```

---

## Credentials (for live agent runs only)

Agent runs use Amazon Bedrock. Credentials are read from the standard AWS chain — **never hard-code or
commit them** (`apieky`, `KEY/`, and `*.csv` key files are git-ignored). Any of these works:

```bash
# environment variables
export AWS_ACCESS_KEY_ID=...        AWS_SECRET_ACCESS_KEY=...   AWS_DEFAULT_REGION=us-east-1
# or a Bedrock API key
export AWS_BEARER_TOKEN_BEDROCK=...
# or a standard ~/.aws/credentials profile
```

Alternatively, place a file named `apieky` at the repo root (AWS console CSV, `KEY=VALUE` dotenv, JSON,
or two bare lines); it is git-ignored. See `trek_agent/credentials.py`.

---

## Data and license

- **Tasks**: `trek_queries.csv` (800) and `trek_gold.jsonl` (800 gold references).
- **Knowledge base**: `api/data/v2/` — 107,195 flights, 39,396 hotels, 55,814 attractions, 10,125 car
  rentals over 375 cities. The released snapshot combines author-curated, de-identified input values,
  synthetic identifiers, and selected calibrated fields; the city/airport scaffold derives from the
  public-domain [OurAirports](https://ourairports.com/data/) dataset. No real travelers' personal data
  or travel histories are included, and evaluation performs no live booking-platform lookup.
- **Results**: `results/scores_summary.json` contains the complete aggregate results;
  `results/scores_summary.csv` and `results/leaderboard.csv` contain the matching CSV summary.
  The 15 `results/trek_*.scores.csv` files contain all 800 per-task scores for each model,
  including the KB-backed D0-key vehicle-capacity check. These scores re-evaluate the existing
  model outputs; the tasks, KB snapshot, and model outputs are unchanged.

Code is released under the **MIT License**; the dataset under **CC BY 4.0**. See [`LICENSE`](LICENSE).
The synthetic prices, schedules, and availability **must not be used as real travel information**.

## Citation

This anonymous snapshot accompanies a manuscript submitted for **double-blind review through
ACL Rolling Review (ARR)**. Author details are withheld in this snapshot; a full citation will be
added after review.
