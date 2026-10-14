# SignalLens: an interactive ML workbench

Upload a CSV, choose what to predict, train real scikit-learn classifiers, and see how well they work, what they rely on and how each prediction responds to its inputs.

> **Repository:** https://github.com/vinuah-dev/signallens · **Live app:** _add your deployed URL here_ · **Demo video:** _add link here_

SignalLens takes you through six steps, shown as a rail on the left of the app:

| # | Stage | What you do | What you see |
|---|-------|-------------|--------------|
| 01 | **Source** | Upload a CSV or load a sample | Inferred column types, missing values, distributions, summary stats, sortable/filterable rows |
| 02 | **Frame** | Choose the target and input features | Live validation, class balance, association scores, leakage/ID/missing-data warnings with one-click fixes |
| 03 | **Train** | Pick an algorithm, tune it, or train all four | Leaderboard and side-by-side metric chart against a naive baseline |
| 04 | **Inspect** | Read one run | Plain-language findings, confusion matrix (click a cell to see those rows), per-class scores, overfitting check, ROC curve, test-row browser |
| 05 | **Explain** | Ask what the model relies on | Permutation importance (with run-vs-run comparison), a "three lenses" rank chart, and an explicit correlation ≠ causation note |
| 06 | **Probe** | Enter feature values | Live prediction and class probabilities, input validation, what moved this prediction, what-if curves, pinned scenarios compared side by side |

Every number in the UI comes from a model trained on the uploaded data. There are no hard-coded outputs.

---

## Quick start (local)

Requirements: Python 3.11+ and Node 18+.

**One command** (from the repo root):

```bash
npm start
```

On Windows you can also double-click `start.cmd` or run `.\start.cmd`. If PowerShell says "running scripts is disabled", use `.\start.cmd` or `npm.cmd start`; that error comes from npm's PowerShell wrapper being blocked by the execution policy. The first run creates `backend/.venv` and installs the Python and npm packages. Every run then starts the API on :8000 and the frontend on :5173 with labelled output. Open http://localhost:5173 and press Ctrl+C to stop both servers. The script is `scripts/dev.mjs`; it uses only Node built-ins.

**Or run the two servers by hand:**

```bash
# 1. API (http://127.0.0.1:8000)
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
uvicorn signallens.api.app:app --reload --port 8000

# 2. Frontend (http://localhost:5173, proxies /api to :8000)
cd frontend
npm install
npm run dev
```

Production-style single server: run `npm run build` in `frontend/`, then start uvicorn. The API serves `frontend/dist` at `/`.

### Tests

```bash
cd backend && pytest          # 99 tests: data validation, framing, training/evaluation, prediction, HTTP API
cd frontend && npm test       # 12 tests: client-side file/number validation, formatting, chart ticks
cd frontend && npm run typecheck
```

---

## Architecture

```
signallens/
├── backend/
│   ├── signallens/
│   │   ├── core/                 ← all ML and data logic; no HTTP, no UI
│   │   │   ├── loading.py        file → DataFrame (format, encoding, delimiter, cleaning)
│   │   │   ├── profiling.py      column kinds, missingness, stats, histograms, row paging
│   │   │   ├── framing.py        target/feature validation, class balance, association, leakage checks
│   │   │   ├── pipeline.py       sklearn Pipeline: cast → impute (+indicators) → scale / one-hot → model
│   │   │   ├── algorithms.py     model registry + hyper-parameter specs (UI form is generated from this)
│   │   │   ├── training.py       split → fit → evaluate → cross-validate → explain
│   │   │   ├── evaluation.py     metrics, baseline, confusion matrix, ROC, plain-language interpretation
│   │   │   ├── explain.py        permutation & native importance, local sensitivity, what-if curves
│   │   │   ├── prediction.py     input validation policy + prediction
│   │   │   ├── store.py          bounded in-memory LRU for datasets and models
│   │   │   ├── config.py         every limit and threshold in one place
│   │   │   └── errors.py         typed errors with stable machine-readable codes
│   │   ├── api/                  ← thin FastAPI layer (routes, request schemas, error mapping, SPA serving)
│   │   └── samples/              bundled CSVs (iris, breast cancer, synthetic churn)
│   ├── scripts/make_samples.py   regenerates the sample CSVs
│   └── tests/                    pytest suite
├── frontend/src/
│   ├── api/                      typed client + response types (mirror of the backend JSON)
│   ├── state/store.tsx           workflow state (useReducer + context), stage availability
│   ├── stages/                   one component per workflow stage
│   ├── charts/                   hand-built SVG charts (histogram, confusion matrix, ROC, importance, lens slope, what-if…)
│   ├── components/ui.tsx         buttons, tooltips, glossary terms, notices, toggles
│   └── lib/                      formatting, client-side validation, glossary, palette
├── docs/TESTING_LOG.md           issues found during testing and how they were fixed
├── scripts/dev.mjs               `npm start`: first-run setup, then API + frontend together
├── Dockerfile                    builds the frontend, serves everything from one container
└── render.yaml                   one-click Render deployment
```

**How the layers are split:**

- `core/` is plain Python and can be tested on its own. It returns JSON-ready dicts and raises `SignalLensError` subclasses with a `code` (e.g. `single_class`, `target_continuous`, `leakage_suspect`).
- `api/` only parses requests, calls `core`, and maps errors to `{"error": {code, message, details}}` with the right HTTP status.
- The frontend never computes ML results. It renders what the API returns and repeats a few checks client-side for instant feedback (file type/size, numeric fields). The server stays the source of truth.
- State is per-process and in memory (bounded LRU). If the server restarts, the UI notices the `*_not_found` codes and offers to start over.

```
Browser (React) ──/api──▶ FastAPI routes ──▶ core.loading ─▶ core.profiling
                                          ├─▶ core.framing  (validation, association)
                                          ├─▶ core.training ─▶ pipeline / evaluation / explain
                                          └─▶ core.prediction (validation ─▶ fitted Pipeline)
```

---

## ML pipeline

1. **Load** (`loading.py`): extension and size checks, binary-file detection, UTF-8/CP1252/Latin-1 decoding, delimiter sniffing (`, ; \t |`). Then:
   - Header names are trimmed and de-duplicated.
   - Blank strings, `NA`, `null`, `?`, `-` and similar become missing values.
   - Numbers written as `$1,200` or `42%` are parsed. If a few cells in a numeric column are junk, they become missing and you get a warning.
   - Fully empty rows and columns are dropped.
   - Files over 200k rows are randomly sampled down.
2. **Profile** (`profiling.py`): each column gets a kind:
   - **numeric**
   - **categorical**
   - **identifier**: unique per row, consecutive integers, or ID-like names
   - **text**: long and mostly unique
   - **datetime**

   Each column also gets missing-value counts, summary stats, a histogram, and flags (constant, high cardinality, integer codes that may really be categories). The likely target is suggested.
3. **Frame** (`framing.py`): validates the question before any training:
   - The target must exist, must not also be a feature, and must not be an ID, text or date.
   - It must have between 2 and 50 classes and must not be a continuous number.
   - Rows with no target value are dropped. So are classes with only one row, which can't appear on both sides of a split.
   - Date features are rejected. Constant, mostly-missing, ID and free-text features get warnings.

   Class imbalance is measured. Each feature's **association** with the target is computed as normalised mutual information. A feature that predicts the target almost perfectly on its own is flagged as **possible leakage**.
4. **Split**: a stratified `train_test_split` (default 80/20, seed 42). If the split is too small to put every class on both sides, you get a clear error instead of a crash.
5. **Preprocess + fit**: one scikit-learn `Pipeline`.
   - `ColumnCaster`: numeric columns become float; categorical values become canonical strings (`3`, `3.0` and `"3"` are the same category).
   - Numeric columns: `SimpleImputer(median, add_indicator=True)` then `StandardScaler`. The indicator keeps "this value was missing" visible to the model.
   - Categorical columns: `SimpleImputer(constant="(missing)")` then `OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30)`.
   - Then one of the classifiers below.

   | Algorithm | Tunable | Imbalance handling |
   |---|---|---|
   | Logistic regression | C (log scale) | `class_weight="balanced"` |
   | Decision tree | max depth, min rows per leaf | `class_weight="balanced"` |
   | Random forest | trees, max depth, min rows per leaf | `class_weight="balanced"` |
   | Gradient boosting (`HistGradientBoostingClassifier`) | learning rate, rounds, max depth | `class_weight="balanced"` |

   **No data leakage:** every learned statistic (medians, scaling, category lists) sits inside the Pipeline. It is fitted on the training split only, then applied unchanged to the test split, the cross-validation folds and user inputs. `test_training.py::test_preprocessing_is_fitted_on_training_rows_only` checks this.
6. **Evaluate** (`evaluation.py`), on the held-out test split:
   - Accuracy, balanced accuracy, and macro- and weighted-averaged precision, recall and F1, plus per-class scores.
   - Confusion matrix, and ROC curve + AUC (binary) or macro one-vs-rest AUC (multiclass).
   - A **most-frequent-class baseline** for context.
   - Train-vs-test scores to spot overfitting.
   - Stratified k-fold **cross-validation** (macro F1) on the training split. Every fold refits the whole Pipeline.

   The scores are turned into short plain-language findings. When classes are imbalanced, the findings lead with balanced accuracy.
7. **Explain** (`explain.py`):
   - **Permutation importance** on the test split, measured as the drop in balanced accuracy over 5 repeats. It is computed on the original columns, so a one-hot encoded category counts as one feature.
   - **Native importance**: tree impurity, or |coefficients|, summed back to the original columns.
   - **Association** from step 3.
   - **Local sensitivity**: how a prediction's probability changes when each feature is reset to its typical training value.
   - **What-if curves**: one feature swept across its training range (1st–99th percentile) while the others stay fixed.

   The UI labels all of these as model behaviour, not causes.
8. **Predict** (`prediction.py`) follows a documented input policy:
   - Non-numeric text in a number field is rejected (422 with the bad fields listed).
   - An empty field is treated as missing, exactly as gaps were handled during training.
   - A value outside the training range is accepted but flagged as extrapolation (marked "far" if it's beyond one full range away). The UI offers to clip it.
   - An unseen category is accepted and flagged.
   - Unknown keys are ignored and noted.

---

## Dataset requirements

- A **CSV** (`.csv`, `.tsv` or `.txt`) with a **header row**, separated by commas, semicolons, tabs or pipes. Max **25 MB**. Larger row counts are sampled down to 200,000 rows.
- At least **10 data rows** and **2 columns**.
- One **target column** with **2–50 repeated classes**. Text labels (`yes`/`no`) or integer codes (`0`/`1`/`2`) both work. Continuous numbers are rejected; bin them first.
- Features can be numeric or categorical, with missing values. ID columns and long free text are detected and left unselected by default. Dates are detected and need to be turned into numeric parts (month, weekday…) first.
- Excel files: export to CSV first. The app says so if you upload an `.xlsx`.

**Bundled samples:**
- **iris**: 3 balanced classes, all numeric.
- **breast_cancer**: binary, 30 numeric features.
- **customer_churn**: synthetic, generated by `scripts/make_samples.py` from a known formula. It has an ID column, a date column, categorical features, missing values, junk tokens in a numeric column, a 21% minority class, and a deliberately leaky `cancellation_reason` column.

---

## Edge cases handled

| Situation | Behaviour |
|---|---|
| Wrong file type, empty file, header only, binary file renamed to `.csv` | Rejected with a specific message (and an Excel hint) |
| Single column / wrong delimiter | "Only one column found — check separators" |
| Ragged rows | "Malformed CSV" with the line number |
| Duplicate headers, blank headers, whitespace, `?`/`NA` tokens | Cleaned and reported |
| Numeric column with a few junk values | Coerced to numbers; junk becomes missing with a warning |
| Target is an ID / text / date / continuous / single class / >50 classes | Rejected with an explanation |
| Rows missing the target, classes with one row | Dropped with a warning |
| Feature that leaks the target | Flagged with a one-click "Drop" |
| Class imbalance | Warning, recommended class balancing, balanced-accuracy headline |
| Split too small for the number of classes | Clear error before training |
| Estimator crash / convergence warning | `training_failed` error / visible warning, server stays up |
| Missing / out-of-range / non-numeric / unseen prediction inputs | Imputed / flagged / rejected / flagged |
| Server restarted (state lost) | UI detects `*_not_found` and offers to start over |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/limits`, `/api/algorithms`, `/api/samples` | Metadata |
| POST | `/api/datasets` (multipart `file`) | Upload + profile |
| POST | `/api/samples/{key}` | Load a bundled dataset |
| GET | `/api/datasets/{id}/rows?offset&limit&missing_only&sort_by&descending` | Page through rows |
| POST | `/api/datasets/{id}/frame` | Validate target/features, get balance, association, warnings |
| POST | `/api/models` | Train + evaluate + explain a model |
| GET | `/api/models/{id}` | Run summary |
| POST | `/api/models/{id}/predict` | Predict 1–100 rows (with notes and sensitivity) |
| POST | `/api/models/{id}/what-if` | Sweep one feature |

Errors always look like `{"error": {"code": "...", "message": "...", "details": {...}}}`.

---

## Testing and refinement

`backend/tests` has **99** tests:
- `test_loading.py`: file validation and cleaning.
- `test_profiling.py`: typing and stats.
- `test_framing.py`: target/feature rules, leakage, imbalance.
- `test_training.py`, covering:
  - Every algorithm learns a separable signal, and random labels score near chance.
  - The split is stratified, and the confusion matrix agrees with accuracy.
  - The baseline is correct, and there is no preprocessing leakage.
  - Results are reproducible, and balancing helps minority recall.
  - Overfitting is detected, and failures are handled.
- `test_prediction.py`: input policy, matching the pipeline, sensitivity, what-if.
- `test_api.py`: status codes, error shape, the full workflow over HTTP.

**Issues found and fixed** (full detail in [`docs/TESTING_LOG.md`](docs/TESTING_LOG.md)):

1. **Mode imputation hid an informative missing value.** A field filled only for churners lost its signal once gaps became the most common reason. Fixed with an explicit `(missing)` category and numeric missing-indicators.
2. **Random forest was 3× slower than necessary.** `n_jobs=-1` made the 150 small predict calls of permutation importance slow (10.2 s → 3.1 s).
3. **"none" was read as a missing value**, erasing a real category ("no internet service").
4. **Duplicate column names were renamed silently** by pandas, so the duplicate warning never fired.
5. **The headline undersold good models on imbalanced data.** A model with balanced accuracy 0.79 and AUC 0.89 was summarised as "barely beats the naive guess". The headline now uses balanced accuracy when classes are imbalanced.

Several UI issues were also fixed during browser testing:
- The ROC axis label overlapped its tick labels.
- The importance axis label had the wrong sign.
- Confusion-matrix shading made small classes invisible.
- The "Drop flagged" count double-counted a column.
- The probe stage didn't detect a server restart.
- On phones, the prediction was off-screen while editing inputs.

---

## AI tools used

<!-- Edit this section in your own words before submitting. -->
This project was built with **Claude Code** (Anthropic's coding agent, Claude Opus 5.5) as a pair programmer. Claude Code:
- Drafted the backend core, the API, the React stages and the hand-written SVG charts.
- Wrote the test suites.
- Drove the app in a browser to test the full workflow.

Refinements came from running the app and its tests rather than accepting the first draft. Each issue above was found by a failing test, by timing, or by using the UI, then fixed and covered by a regression test where possible. Design decisions were chosen deliberately, and the README explains them:
- Explicit missing-value handling.
- Permutation importance computed on the original columns.
- A balanced-accuracy headline for imbalanced data.
- A rank-based "three lenses" chart instead of mixing incompatible units.

---

## Deployment

The `Dockerfile` builds the frontend and runs everything in one container on `$PORT`.

- **Render** (free): push the repo to GitHub, then *New → Blueprint* and pick the repo (`render.yaml` is included). Or create a *Web Service* with runtime *Docker*.
- **Hugging Face Spaces**: create a *Docker* Space, push the repo, and set the variable `PORT=7860`.
- **Any Docker host**: `docker build -t signallens . && docker run -p 8000:8000 signallens`.

Free tiers sleep when idle. The first request can take ~30 s, and sleeping clears in-memory state. The UI handles both cases.

If you host the frontend separately (e.g. Vercel), build it with `VITE_API_BASE=https://your-api` and set `CORS_ORIGINS=https://your-frontend` on the API.

---

## Limitations and next steps

- State is in memory and per process. A multi-instance deployment would need Redis or disk-backed storage, plus model persistence (joblib).
- Classification only. Regression and time-aware splits are out of scope.
- Local explanations reset one feature at a time. They don't capture interactions the way SHAP values do.
- No batch prediction upload yet. The API already accepts up to 100 rows per call.

## Tech stack

- **Backend:** Python 3.12, FastAPI, scikit-learn 1.6, pandas, NumPy, pytest.
- **Frontend:** React 18, TypeScript, Vite, hand-built SVG charts (no chart library), Vitest. Fonts: Fraunces, IBM Plex Sans/Mono (self-hosted via Fontsource).
- **Deploy:** Docker (single container), Render blueprint.
