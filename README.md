## Hospital Readmission Risk Dashboard

Helping clinical teams spot high-risk diabetic patients *before* they bounce back through the hospital doors. This project stitches together richer encounter data, sharper features, and an interactive Streamlit experience to deliver readmission insights in minutes.

---

### What We Shipped
- **Merged intelligence** – both `diabetic_data.csv` and the expanded `diabetic_data2.csv` now flow through a single cleaner, deduplicated pipeline that drops hospice/expired encounters automatically.
- **Feature engineering spree** – age midpoints, comorbidity counts, medication intensity scores, emergency-source flags, and more—all derived from ICD-9 and medication change signals with zero manual prep.
- **Model showdown** – Logistic Regression stays as a reference, while XGBoost leads the pack (precision 0.67 @ threshold 0.90 in the latest run, still shy of the 0.75 goal—ripe for tuning).
- **Dashboard glow-up** – the Streamlit interface groups inputs into Patient Snapshot, Visit Details, Utilization History, and Treatments & Labs, with an optional diagnoses expander for power users.

---

### Run the Workflow
```bash
# 1. Spin up a virtual environment (Python 3.11+ recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / CMD
# source .venv/bin/activate     # macOS / Linux shells

# 2. Install project dependencies
pip install -r requirements.txt

# 3. Engineer features (defaults to both bundled raw CSVs)
python src/data_preprocessing.py

# 4. Train & evaluate models (writes artifacts to /models and /reports)
python src/train_model.py

# 5. Launch the interactive dashboard
python -m streamlit run app/streamlit_app.py
```
The app opens at `http://localhost:8501`. Dial the threshold slider, enter patient details, and review readmission probabilities alongside the latest evaluation metrics.

---

### Project Structure
```
AI-dvp-workflow/
├── app/
│   └── streamlit_app.py          # Streamlit UI for risk exploration
├── data/
│   ├── raw/                      # Source CSVs (diabetic_data*.csv, IDS_mapping.csv)
│   └── processed/                # Engineered dataset (diabetic_data_processed.csv)
├── models/
│   └── readmission_xgb_pipeline.pkl
├── reports/
│   ├── metrics.json              # Test-set metrics for the best model
│   ├── model_metrics.json        # Per-model threshold summaries
│   ├── threshold_metrics.csv     # Threshold sweep (0.05 → 0.95)
│   └── confusion_matrix.png      # Saved confusion matrix plot
├── scripts/
│   └── explore_thresholds.py     # Optional threshold exploration helper
├── src/
│   ├── data_preprocessing.py     # CLI feature engineering pipeline
│   └── train_model.py            # Model training, evaluation, artifact export
└── requirements.txt              # Dependency pins for reproducibility
```

---

### Performance Snapshot
```json
{
  "model": "xgboost",
  "threshold": 0.90,
  "test_precision": 0.67,
  "test_recall": 0.0035,
  "test_accuracy": 0.887,
  "scale_pos_weight": 7.81,
  "meets_constraints": false
}
```
Takeaways:
- High precision but under the 0.75 target—the slider now stretches to `0.95`, so experiment with lower cutoffs or rebalance class weights to lift recall.
- The current operating point prioritizes avoiding false positives (recall ~0.35%). Adjust to fit clinical tolerance.
- Logistic Regression is still trained for comparison and can be tuned further (watch for `ConvergenceWarning`—raise `max_iter` if needed).

---

### Customize & Extend
- **Bring your own data** – chain multiple `--input-path` arguments for `src/data_preprocessing.py` to include site-specific extracts.
- **Tweak the models** – extend `build_model_specs` in `src/train_model.py` with new estimators or hyperparameters.
- **Refresh the UI** – the Streamlit sections are modular; surface additional vitals, charts, or explanations quickly.
- **Dive into thresholds** – use `reports/threshold_metrics.csv` or `scripts/explore_thresholds.py` to pick an operating point that fits your precision/recall trade-offs.

---

### Contact
For questions, ideas, or contributions, open an issue or ping the project maintainer.

---

### Group Members
- Mary Wairimu
- Kelvin Karani
- Fred Kaloki
- Odii Chineye Gift
- Rivaldo Ouma

Let’s keep patients out of unnecessary readmissions—happy experimenting!