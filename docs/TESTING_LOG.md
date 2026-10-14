# Testing & refinement log

Issues found while testing SignalLens, how each was found, and the fix. Each
fix has a regression test in `backend/tests` (or the frontend tests) where
it can be tested automatically.

## 1. Mode imputation hid an informative "missing" signal

- **Found by:** smoke-testing the synthetic churn dataset. The leakage
  detector flagged `cancellation_reason` (only filled in for churned customers)
  as almost perfectly predictive, yet a model trained *with* it only reached
  ~91% accuracy instead of ~100%.
- **Cause:** categorical gaps were filled with the most frequent category. For
  non-churners the empty reason became `"price"` — the most common reason
  among churners — so the model could no longer tell "no reason" from "price".
  Missingness itself carried the signal, and imputation erased it.
- **Fix:** categorical gaps now become an explicit `(missing)` category, and
  numeric gaps get the training median **plus** a 0/1 missing-indicator column
  (`SimpleImputer(add_indicator=True)`). Importances map indicator columns back
  to their source feature.
- **Test:** `test_training.py::test_missingness_is_visible_to_the_model`.

## 2. Random forest was 3× slower because of parallelism

- **Found by:** timing each training stage on the breast-cancer sample (RF
  run took ~14 s end to end).
- **Cause:** `n_jobs=-1` made fitting slightly faster but every *prediction*
  spun up worker threads. Permutation importance makes 150 small predict calls
  (30 features × 5 repeats), so it went from 3.1 s to 10.2 s.
- **Fix:** random forest runs single-threaded (`n_jobs` default).

## 3. The word "none" was being read as a missing value

- **Found by:** inspecting the churn profile: `internet_service` showed ~18%
  missing and no `none` category, although the generator writes `none` for
  customers without internet.
- **Cause:** the loader's extra missing-value tokens included `none`/`None`,
  so a legitimate category was silently erased (329 "missing" vs 72 real gaps).
- **Fix:** extra tokens reduced to `?`, `-`, `--`; pandas' own defaults
  (`NA`, `N/A`, `null`, empty…) still apply.
- **Test:** `test_loading.py::test_none_is_a_real_category`.

## 4. Duplicate column names were renamed silently

- **Found by:** `test_loading.py::test_duplicate_columns_renamed` failed on
  its first run: columns came back as `a, a.1` and no warning was emitted.
- **Cause:** pandas de-duplicates headers itself (`a` → `a.1`) before our own
  de-duplication ran, so our check never saw a duplicate and never warned.
- **Fix:** the loader re-reads the raw header row with the detected delimiter
  and applies its own renaming (`a_1`), which also records a warning shown in
  the UI.

## 5. The headline undersold good models on imbalanced data

- **Found by:** walking through the churn workflow in the browser. A balanced
  logistic regression (balanced accuracy 0.79, ROC AUC 0.89, catching 79% of
  churners) was summarised as *"barely beats the naive guess"* with a warning,
  because its plain accuracy (79.4%) was close to always guessing "No" (78.8%).
- **Cause:** both the generated interpretation and the Inspect headline
  compared plain accuracy with the majority-class baseline. On imbalanced data
  that baseline is high *because* it ignores the minority class, and class
  balancing deliberately trades a little accuracy for minority recall.
- **Fix:** when the frame is flagged as imbalanced, the first finding and the
  big headline number compare **balanced accuracy** (naive guess = 1 / number
  of classes) and say explicitly that the majority guess never finds any other
  class. Balanced data keeps the plain-accuracy wording.
- **Test:** `test_training.py::test_imbalanced_headline_uses_balanced_accuracy`.
