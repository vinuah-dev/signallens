# Demo script (~3 minutes)

A walkthrough for recording the required demo, using the bundled **customer churn** sample because it exercises every part of the pipeline.

1. **Source** (30 s)
   - Click **Load** on "Customer churn (synthetic)".
   - Point out the ledger: 1,800 rows, 12 columns, 7.7% missing cells.
   - Show the warning that `total_charges` had 12 junk values ("unknown") that became missing.
   - In *Columns*, note `customer_id` (ID) and `signup_date` (date) were detected. Click `monthly_charges` to show its histogram and stats.
   - In *Rows*, toggle **Only rows with gaps** and sort a column.
2. **Error handling** (15 s, optional)
   - Click **Use a different file** and drop an `.xlsx` or an empty `.csv` to show the specific error.
   - Reload the sample.
3. **Frame** (30 s)
   - `churned` is pre-selected as the target.
   - In the readiness panel, show the class balance (79/21, "imbalanced").
   - Show the leakage warning on `cancellation_reason` (100% association) and click **Drop**.
   - Mention that the ID and date columns were left out automatically.
4. **Train** (30 s)
   - Leave **Balance classes** on (it's recommended because the classes are imbalanced).
   - Click **Train all 4 with defaults**.
   - Show the leaderboard, the best-score underlines, and the *Side by side* chart with the grey naive-baseline ticks.
5. **Inspect** (40 s)
   - Select logistic regression.
   - Read the headline: balanced accuracy vs 50% for the naive guess, and why plain accuracy would mislead here.
   - Click the **No → Yes** cell of the confusion matrix to filter the test rows to those mistakes.
   - Show the fit check (train ≈ CV ≈ test means no overfitting) and the ROC curve hover.
6. **Explain** (30 s)
   - Show permutation importance: `contract` matters most.
   - Use **compare with** to overlay the random forest.
   - Show the three-lenses chart: `internet_service` ranks high inside the model but low on unseen rows.
   - Read the causation note.
7. **Probe** (40 s)
   - Back in Inspect, press **Probe →** on a test row; the real label is shown.
   - Change `contract` to "two-year" and watch the probability fall.
   - Show *What moved this prediction* and the *What if…* sweep for `tenure_months`.
   - Type `500` in tenure to show the out-of-range warning and **clip to range**. Type `abc` to show the validation error.
   - Pin two scenarios and compare them.
