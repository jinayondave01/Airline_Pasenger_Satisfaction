"""Flask backend for the Airline Passenger Satisfaction Random Forest model.
 
Endpoints
---------
GET  /health         -> liveness + model metadata
GET  /api/options    -> dropdown options / numeric ranges for building the form
POST /api/predict    -> {"prediction", "label", "probability_satisfied", ...}
POST /api/explain    -> same as predict + top SHAP feature contributions
 
Run (dev):   python app.py
Run (prod):  gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 app:app   # 1 worker: the model + SHAP explainer are held in memory
"""
import json
import logging
import os
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "airline_model_pipeline.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
INFO_PATH = BASE_DIR / "model_info.json"
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("airline-api")
 
app = Flask(__name__)
app.json.sort_keys = False                  
 

model = joblib.load(MODEL_PATH)               
FEATURES = joblib.load(FEATURES_PATH)         
with open(INFO_PATH) as f:
    INFO = json.load(f)
 
CATEGORICAL = INFO["categorical_options"]     
RATING_COLS = set(INFO["rating_columns"])     
LABELS = {int(k): v for k, v in INFO["target_mapping"].items()}
 

NUMERIC_BOUNDS = {
    "Age": (0, 120),
    "Flight Distance": (1, 20000),
    "Departure Delay": (0, 3000),
    "Arrival Delay": (0, 3000),               
}
OPTIONAL = {"Arrival Delay"}                  


TRAIN_RANGES = INFO["numeric_ranges"]
MIN_ZERO_ROWS = 100
ZERO_COUNTS = INFO.get("rating_zero_counts") or {
    "Departure and Arrival Time Convenience": 6681, "Ease of Online Booking": 5682,
    "In-flight Wifi Service": 3916, "Online Boarding": 3080, "Leg Room Service": 598,
    "Food and Drink": 132, "In-flight Entertainment": 18, "Cleanliness": 14,
    "On-board Service": 5, "In-flight Service": 5, "Check-in Service": 1,
    "Gate Location": 1, "Seat Comfort": 1, "Baggage Handling": 0,
}


def _rating_bounds(col):
    low, high = int(TRAIN_RANGES[col][0]), int(TRAIN_RANGES[col][1])
    if low == 0 and ZERO_COUNTS.get(col, 0) < MIN_ZERO_ROWS:
        low = 1                                   
    return low, high


RATING_BOUNDS = {c: _rating_bounds(c) for c in RATING_COLS}
 

_explainer = None
_feature_names_out = None
try:
    import shap
 
    _prep = model.named_steps["prep"]
    _rf = model.named_steps["clf"]
    _feature_names_out = [n.split("__", 1)[1] for n in _prep.get_feature_names_out()]
    _explainer = shap.TreeExplainer(_rf)
    log.info("SHAP explainer ready")
except Exception as exc:  
    log.warning("SHAP unavailable, /api/explain disabled: %s", exc)
 
 

class ValidationError(ValueError):
    pass
 
 
def validate_and_build_row(payload):
    """Validate a JSON dict. Returns (one-row DataFrame in training order, list of warnings)."""
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")
 
    errors, row, notes = [], {}, []
    for col in FEATURES:
        value = payload.get(col)
 
        if value is None:
            if col in OPTIONAL:
                row[col] = np.nan
            else:
                errors.append(f"'{col}' is required.")
            continue
 
        if col in CATEGORICAL:
            if value not in CATEGORICAL[col]:
                errors.append(f"'{col}' must be one of {CATEGORICAL[col]}.")
            else:
                row[col] = value
            continue
 
        # numeric
        if isinstance(value, bool):
            errors.append(f"'{col}' must be a number.")
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a number.")
            continue
        if not np.isfinite(num):
            errors.append(f"'{col}' must be a finite number.")
            continue
 
        if col in RATING_COLS:
            lo, hi = RATING_BOUNDS[col]
            if num != int(num) or not lo <= num <= hi:
                errors.append(f"'{col}' must be an integer from {lo} to {hi}.")
                continue
            row[col] = int(num)
        else:
            lo, hi = NUMERIC_BOUNDS.get(col, (-np.inf, np.inf))
            if not lo <= num <= hi:
                errors.append(f"'{col}' must be between {lo} and {hi}.")
                continue
            t_lo, t_hi = TRAIN_RANGES.get(col, (-np.inf, np.inf))
            if not t_lo <= num <= t_hi:
                notes.append(f"{col} = {num:g} is outside the range the model was trained on "
                             f"({t_lo:g} to {t_hi:g}); treat this result with caution.")
            row[col] = num
 
    unknown = sorted(set(payload) - set(FEATURES))
    if unknown:
        errors.append(f"Unknown fields: {unknown}.")
    if errors:
        raise ValidationError(errors)
 
    return pd.DataFrame([row])[FEATURES], notes
 
 
def run_prediction(df):
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "label": LABELS[pred],
        "probability_satisfied": round(proba, 4),
        "probability_neutral_or_dissatisfied": round(1 - proba, 4),
    }
 
 
def run_explanation(df, top_n=8):
    """SHAP contribution per ORIGINAL feature. One-hot columns (e.g. Class_Business,
    Class_Economy) are summed back into one entry ("Class") so the page never shows a
    category the passenger did not choose."""
    X_t = model.named_steps["prep"].transform(df)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    sv_all = _explainer(pd.DataFrame(X_t, columns=_feature_names_out)).values
    sv = sv_all[0, :, 1] if sv_all.ndim == 3 else sv_all[0]
    base = float(np.ravel(_explainer.expected_value)[-1])

    contrib = {}
    for name, val in zip(_feature_names_out, sv):
        original = next((c for c in CATEGORICAL if name.startswith(c + "_")), name)
        contrib[original] = contrib.get(original, 0.0) + float(val)
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    raw = df.iloc[0]

    def shown(col):                                   
        v = raw[col]
        if pd.isna(v):
            return None
        return v.item() if hasattr(v, "item") else v

    return {
        "base_value": round(base, 4),
        "contributions": [
            {
                "feature": name,
                "value": shown(name),
                "shap": round(val, 4),
                "direction": "toward Satisfied" if val > 0 else "toward Neutral/Dissatisfied",
            }
            for name, val in ranked
        ],
    }



ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
 
 
@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin and ("*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS):
        resp.headers["Access-Control-Allow-Origin"] = "*" if "*" in ALLOWED_ORIGINS else origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp
 
 

def _fmt(metrics, keys, pct=True):
    """Return the first available metric (tries several likely key names)."""
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            v = v * 100 if (pct and v <= 1) else v
            return f"{v:.2f}%" if pct else f"{v:.3f}"
    return "n/a"
 
 
@app.get("/")
def home():
    m = INFO.get("metrics") or {}
    metrics = {
        "test_accuracy_pct": _fmt(m, ["test_accuracy", "accuracy", "test_acc"]),
        "cv_accuracy_pct": _fmt(m, ["cv_accuracy", "cv_accuracy_mean", "cv_mean", "cv_acc"]),
        "roc_auc": _fmt(m, ["roc_auc", "test_roc_auc", "auc"], pct=False),
    }
    return render_template("index.html", metrics=metrics)
 
 
@app.get("/predict")
def predict_page():
    ratings = [c for c in FEATURES if c in RATING_COLS]   
    rating_min = {c: RATING_BOUNDS[c][0] for c in ratings}
    return render_template("predict.html", categorical=CATEGORICAL, ratings=ratings, rating_min=rating_min)
 
 
@app.get("/result")
def result_page():
    return render_template("result.html")
 
 
@app.get("/health")
def health():
    return jsonify(status="ok", metrics=INFO.get("metrics"), versions=INFO.get("versions"),
                   explain_enabled=_explainer is not None)
 
 
@app.get("/api/options")
def options():
    return jsonify(
        features=FEATURES,
        categorical_options=CATEGORICAL,
        rating_columns=sorted(RATING_COLS),
        rating_bounds={c: {"min": lo, "max": hi} for c, (lo, hi) in RATING_BOUNDS.items()},
        rating_note="0 = not applicable. Offered only where it is common enough in the training data.",
        numeric_bounds={k: {"min": lo, "max": hi} for k, (lo, hi) in NUMERIC_BOUNDS.items()},
        optional_fields=sorted(OPTIONAL),
    )
 
 
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    try:
        df, notes = validate_and_build_row(request.get_json(silent=True))
    except ValidationError as exc:
        return jsonify(error="Invalid input", details=exc.args[0]), 400
    return jsonify({**run_prediction(df), "warnings": notes})
 
 
@app.route("/api/explain", methods=["POST", "OPTIONS"])
def explain():
    if request.method == "OPTIONS":
        return "", 204
    if _explainer is None:
        return jsonify(error="SHAP is not installed on the server."), 501
    try:
        df, notes = validate_and_build_row(request.get_json(silent=True))
    except ValidationError as exc:
        return jsonify(error="Invalid input", details=exc.args[0]), 400
    return jsonify({**run_prediction(df), "warnings": notes, **run_explanation(df)})
 
 
@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found"), 404
 
 
@app.errorhandler(500)
def server_error(exc):
    log.exception("Unhandled error: %s", exc)
    return jsonify(error="Internal server error"), 500
 
 
if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)