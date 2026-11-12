# Airline Passenger Satisfaction

### A Web Application for Predicting and Explaining Flight Passenger Satisfaction

A Flask web application that uses a **Random Forest classifier** to classify a passenger's overall satisfaction as **Satisfied** or **Neutral or Dissatisfied**. Users enter a passenger profile, flight details, and service ratings; the application returns a prediction, its estimated probability, and a **SHAP explanation** of the most influential input features.

> **Scope:** This is a **post-flight survey classification** tool. Because many inputs are service-experience ratings, it does **not** forecast a passenger's satisfaction before taking a flight.

## Key Features

- **Passenger input form** with 22 inputs: personal details, flight information, and 14 service ratings.
- **Satisfaction classification** with a predicted probability of being satisfied.
- **Explainable AI** using SHAP to show the top eight features affecting a particular prediction.
- **Input validation** for required fields, categorical selections, rating ranges, and numeric values.
- **Missing arrival delay handling**: an optional Arrival Delay is filled using the training-set median by the saved preprocessing pipeline.
- **Model performance dashboard** with held-out test accuracy, cross-validation accuracy, and ROC-AUC.
- **Flask JSON endpoints** for prediction, explanations, model health, and available input options.

## Model and Dataset

| Item | Description |
| --- | --- |
| Machine learning task | Supervised binary classification |
| Algorithm | Random Forest Classifier |
| Dataset | 129,880 airline passenger records |
| Original columns | 24: passenger ID, 22 input features, and satisfaction target |
| Input features | 4 categorical, 4 continuous numeric, and 14 ordinal service ratings |
| Train/test split | 80% / 20%, stratified; `random_state=42` |
| Preprocessing | One-hot encoding for categoricals; median imputation for numeric inputs |
| Random Forest settings | `n_estimators=150`, `max_depth=20`, `min_samples_leaf=3`, `random_state=42` |
| Explainability | SHAP TreeExplainer |

The `ID` column is excluded from training. The target is encoded as `1` for **Satisfied** and `0` for **Neutral or Dissatisfied**. The model's preprocessing and classifier are saved together in a scikit-learn `Pipeline`, so the web application applies the same transformations used during training.

### Reported Evaluation Results

These are the recorded results from the project's training and documentation, **not new benchmark runs performed by this README**.

| Metric | Result |
| --- | ---: |
| Held-out test accuracy | **96.33%** |
| Training accuracy | 97.94% |
| Mean 5-fold cross-validation accuracy | **96.25%** |
| ROC-AUC | **0.9942** |
| Held-out test size | 25,976 passengers |

The two target classes are not perfectly balanced: approximately **56.6% Neutral or Dissatisfied** and **43.4% Satisfied** in the dataset.

## Technology Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript (Flask/Jinja templates)
- **Machine learning:** scikit-learn, pandas, NumPy
- **Explainability:** SHAP
- **Model files:** Joblib `.pkl` artifacts
- **Development and analysis:** VS Code and a Jupyter/Google Colab notebook

The serialized model records Python **3.13.18** and scikit-learn **1.9.1** in `model_info.json`. Use an environment compatible with the versions in `requirements.txt` when loading the model.

## Project Structure

```text
Airline_Pasenger_Satisfaction/
├── app.py                              # Flask app, validation, prediction, SHAP and API routes
├── airline_model_pipeline.pkl          # Trained preprocessing + classifier pipeline
├── feature_names.pkl                   # Input feature names/order
├── model_info.json                     # Categories, ranges, labels and reported metrics
├── requirements.txt                    # App dependencies
├── test_api.py                         # Flask API smoke tests
├── templates/
│   ├── base.html
│   ├── index.html                      # Home/performance summary
│   ├── predict.html                    # Passenger input form
│   └── result.html                     # Prediction and explanation output
├── static/
│   └── style.css
├── Airline_Passenger_Satisfaction_fixed.ipynb  # Model training and evaluation notebook
└── README.md
```

**Dataset note:** The training notebook expects `airline_passenger_satisfaction.csv` in its working directory. The CSV is a separate input file and is **not required to run the web app**, because the application loads the already-trained `.pkl` pipeline. If the CSV is not included in this repository, obtain the matching dataset and place it beside the notebook before rerunning training.

Dataset source cited in the project documentation: [Airline Passenger Satisfaction (Kaggle)](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).

## Run the Web Application Locally (Windows / VS Code)

1. **Clone or download** this repository and open the folder in VS Code.
2. Open a PowerShell terminal in the folder containing `app.py`.
3. Create a virtual environment with a compatible Python version:

   ```powershell
   py -3.13 -m venv .venv
   ```

   If `py -3.13` is unavailable, install a compatible Python version first; do not rename or remove the trained model files.

4. Install dependencies using the environment's Python executable:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

6. Open **http://127.0.0.1:5000** in a browser. Select **Start a prediction**, complete the passenger form, and submit to see the satisfaction result and SHAP feature contributions.

> **Note:** The development server runs locally. Uploading source code to GitHub does not automatically deploy the Flask application. The `gunicorn` dependency is for compatible Unix-like deployment environments, not native Windows serving.

### Run Smoke Tests

From the same project folder, with dependencies installed:

```powershell
.\.venv\Scripts\python.exe test_api.py
```

The test script checks model health, prediction and explanation endpoints, validation of inputs, and handling of optional Arrival Delay. These tests exercise the Flask application without manually starting its server.

## How Prediction Works

1. The user enters **22 passenger and flight inputs** in the web form.
2. `app.py` checks that required values, categories, and numeric ranges are valid.
3. The saved scikit-learn pipeline **encodes categories** and **imputes missing numeric values**.
4. The Random Forest produces the probability of **Satisfied**. A value of **0.5 or above** is classified as Satisfied.
5. For explanations, **SHAP TreeExplainer** calculates feature contributions. One-hot encoded categorical contributions are grouped back under their original feature names, and the **top eight** are shown.

### Main Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Home and model metrics |
| `/predict` | GET | Passenger input form |
| `/result` | GET | Result view |
| `/health` | GET | Model status and metadata |
| `/api/options` | GET | Feature names, dropdown choices, and input limits |
| `/api/predict` | POST | Validate JSON and return classification probabilities |
| `/api/explain` | POST | Return prediction plus SHAP feature contributions |

The `/api/explain` route depends on SHAP being installed and the explainer initializing successfully.

## Reproduce Model Training (Notebook)

1. Open `Airline_Passenger_Satisfaction_fixed.ipynb` in Jupyter Notebook, VS Code's notebook editor, or Google Colab.
2. Put the matching `airline_passenger_satisfaction.csv` in the notebook's working directory.
3. Install the notebook's required dependencies as appropriate for your environment, then run the cells in order.
4. The notebook inspects the dataset, prepares the features and target, trains and evaluates the pipeline, produces plots, runs SHAP explanations, and saves these artifacts:

   - `airline_model_pipeline.pkl`
   - `feature_names.pkl`
   - `model_info.json`

**Caution:** Retraining and overwriting the existing model files will change the web application's prediction behavior unless the training code, data, and package versions are consistent. Keep a backup of the working artifacts before regenerating them.

## Limitations

- Results describe **patterns learned from post-flight survey data**, not guaranteed satisfaction for any individual traveler.
- **SHAP measures contributions to the model prediction, not cause-and-effect relationships** or proof that changing a service alone would change the real-world outcome.
- Generalization to other airlines, routes, or time periods has not been established by the provided dataset.
- Age and gender are included in the inputs; fairness should be reviewed before using the system in operational decisions.
- The web interface evaluates one passenger at a time and does not currently provide bulk CSV scoring.

## Academic Project

**Course:** CST 9/L — CS Professional Track 3  
**Group:** 11 — The Elders  
**Institution:** University of Mindanao Tagum College

**Project members:**
- El David Gyle Jinayon
- Stephen Glenn Maynocas
- Charlie James Buno

This repository is an academic machine learning demonstration for predicting and explaining airline passenger satisfaction from passenger and service survey data.
